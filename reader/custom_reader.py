import os.path as p
import random

import numpy as np
import torch
from torch import nn
from trainer_qa import QuestionAnsweringTrainer

from transformers.modeling_outputs import QuestionAnsweringModelOutput
from reader.base_reader import BaseReader
from reader.custom_head import (
    DprQAHead,
    LstmQAHead,
    CnnQAHead,
    ComplexCnnQAHead,
    ComplexCnnQAHead_v2,
    CnnLstmQAHead,
    ComplexCnnEmQAHead,
    ComplexCnnLstmEmQAHead
)

READER_HEAD = {
    "DPR": DprQAHead,
    "LSTM": LstmQAHead,
    "CNN": CnnQAHead,
    "CCNN": ComplexCnnQAHead,
    "CCNN_v2": ComplexCnnQAHead_v2,
    "CNN_LSTM": CnnLstmQAHead,
    "CCNN_EM": ComplexCnnEmQAHead,
    "CCNN_LSTM_EM": ComplexCnnLstmEmQAHead
}


class CustomModel(nn.Module):
    def __init__(
        self, 
        backbone, 
        head, 
        input_size, 
        pooling_pos, 
        masking_ratio, 
        special_token_ids, # for random masking
        mask_token_id, # for random masking
        freeze_backbone
    ):
        super().__init__()
        self.backbone = backbone

        self.qa_outputs = READER_HEAD[head](input_size=input_size)
        self.qa_outputs.apply(self._init_weight)

        self.head = head
        self.pooling_pos = pooling_pos
        self.masking_ratio = masking_ratio
        self.special_token_ids = special_token_ids
        self.mask_token_id = mask_token_id

        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                param.requires_grad = False

    def _init_weight(self, model):
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            if isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if "weight" in name:
                        nn.init.kaiming_uniform_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)

    def random_masking(self, input_ids):
        ratio = self.masking_ratio / 2 # span masking two tokens at a time
        masked_input_ids = input_ids.clone()

        for input_id in masked_input_ids:
            masked_num = 0

            while masked_num < int(len(input_id) * ratio):
                target_pos = random.randrange(len(input_id) - 1)
                if input_id[target_pos] not in self.special_tokens_ids:
                    input_id[target_pos] = self.mask_token_id
                    if input_id[target_pos + 1] not in self.special_tokens_ids:  
                        input_id[target_pos + 1] = self.mask_token_id
                    masked_num += 1

        return masked_input_ids

    def get_exact_match_token(self, input_batch):
        exact_match_token = []
        for input_ids in input_batch.cpu().numpy():
            is_question = True
            match_token = [0] * len(input_ids)
            question_token_set = set()

            for idx, input_id in enumerate(input_ids[:-1]):
                if is_question and input_id == 3:
                    is_question = False
                    continue
                elif not is_question and input_id == 3:
                    break

                if is_question:
                    question_token_set.add(input_id)
                elif not is_question and (input_id in question_token_set):
                    match_token[idx] = idx

            exact_match_token.append(match_token)

        exact_match_token = torch.Tensor(np.array(exact_match_token)).long().cuda()
        return exact_match_token

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        outputs = self.backbone(
            self.random_masking(input_ids) if (self.training and self.masking_ratio != 0.0) else input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]

        logits = None
        if self.head == "CCNN_LSTM_EM" or self.head == "CCNN_EM":
            exact_match_token = self.get_exact_match_token(input_ids)
            logits = self.qa_outputs((sequence_output, exact_match_token))
        else:
            logits = self.qa_outputs(sequence_output)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[self.pooling_pos :]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class EvalCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if args.do_eval_during_training and state.global_step % args.eval_step == 0:
            control.should_evaluate = True

        return control


class CustomHeadReader(BaseReader):
    def __init__(self, args, model, tokenizer, eval_answers):
        super().__init__(args, None, tokenizer, eval_answers)
        self.model = CustomModel(
            backbone=model,
            head=args.model.reader_name,
            input_size=model.config.hidden_size,
            pooling_pos=2 if "bert" in args.model.model_name_or_path else 1,
            masking_ratio=args.train.masking_ratio,
            special_token_ids=tokenizer.all_special_ids,
            mask_token_id=tokenizer.mask_token_id,
            freeze_backbone=args.train.freeze_backbone,
        )

        if p.exists(p.join(args.model.model_path, "pytorch_model.bin")):  # for checkpoint loading
            self.model.load_state_dict(torch.load(p.join(args.model.model_path, "pytorch_model.bin")))

    def get_trainer(self):
        trainer = QuestionAnsweringTrainer(
            model=self.model,
            args=self.args.train,  # training_args
            custom_args=self.args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            eval_examples=self.eval_examples,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            post_process_function=self._post_processing_function,
            compute_metrics=self._compute_metrics,
            callbacks=[EvalCallback],
        )

        return trainer
