import random
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from trainer_qa import QuestionAnsweringTrainer

from transformers.modeling_outputs import QuestionAnsweringModelOutput
from reader.base_reader import BaseReader, EvalCallback
from reader.custom_head import LstmQAHead, CnnQAHead, FcQAHead, ComplexCnnQAHead, CnnLstmQAHead

READER_HEAD = {"LSTM": LstmQAHead, "CNN": CnnQAHead, "FC": FcQAHead, "CCNN": ComplexCnnQAHead, "CNN_LSTM": CnnLstmQAHead}

class CustomModel(nn.Module):
    def __init__(self, backbone, head, pooling_pos, masking_ratio, freeze_backbone):
        super().__init__()
        self.backbone = backbone

        head_input_size = 768 # 현재 embedding 768 기준, xlm-roberta-large의 경우 1024
        self.qa_outputs = READER_HEAD[head](input_size=head_input_size)
        self.qa_outputs.apply(self._init_weight)

        self.head = head
        self.pooling_pos = pooling_pos
        self.masking_ratio = masking_ratio

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
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias) 

    def random_masking(self, input_ids): #ratio - masking비율
        # BERT 모델의 일반적인 토큰 ID 기준
        ratio = self.masking_ratio / 2
        masked_input_ids = input_ids.clone()
        
        PAD_TOKEN_ID = 0
        CLS_TOKEN_ID = 2
        SEP_TOKEN_ID = 3
        MASK_TOKEN_ID = 4
        except_token = [PAD_TOKEN_ID, CLS_TOKEN_ID, SEP_TOKEN_ID, MASK_TOKEN_ID]

        for input_id in masked_input_ids:
            masked_num = 0
            
            while masked_num < int(len(input_id) * ratio):
                target_pos = random.randrange(len(input_id) - 1)
                if input_id[target_pos] not in except_token:
                    input_id[target_pos] = MASK_TOKEN_ID
                    if input_id[target_pos + 1] not in except_token: #연속 단어 마스킹 가능하면 ㄱ
                        input_id[target_pos + 1] = MASK_TOKEN_ID
                    masked_num += 1
        
        return masked_input_ids

    def get_exact_match_token(self, input_batch):
        exact_match_token = []
        for input_ids in input_batch:
            is_question = True
            match_token = [0] * len(input_ids)
            question_dict = defaultdict(int)

            for idx, input_id in enumerate(input_ids[:-1]):
                if input_id == 0:
                    break
                    
                if is_question and input_id == 3:
                    is_question = False
                    continue
                
                if is_question:
                    question_dict[input_id] += 1
                elif not is_question and question_dict[input_id]:
                    match_token[idx] = idx
            exact_match_token.append(match_token)
        
        exact_match_token = torch.LongTensor(np.array(exact_match_token)).cuda()
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
        if self.head == 'CNN_LSTM':
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
            output = (
                start_logits,
                end_logits,
            ) + outputs[self.pooling_pos:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class CustomHeadReader(BaseReader):
    def __init__(self, args, model, tokenizer, eval_answers):
        super().__init__(args, None, tokenizer, eval_answers)
        self.model = CustomModel(
            backbone=model,
            head=args.model.reader_name, 
            pooling_pos=2 if 'bert' in args.model.model_name_or_path else 1,
            masking_ratio=args.train.masking_ratio,
            freeze_backbone=False
        )

        if args.model.model_path != "": # for checkpoint loading
            self.model.load_state_dict(torch.load(args.model.model_path))

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
            callbacks=[EvalCallback]
        )

        return trainer
