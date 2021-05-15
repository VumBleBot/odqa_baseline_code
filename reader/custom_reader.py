import random

import numpy as np
import torch
from torch import nn
from trainer_qa import QuestionAnsweringTrainer

from transformers.modeling_outputs import QuestionAnsweringModelOutput
from reader.base_reader import BaseReader
from reader.custom_head import LstmQAHead, CnnQAHead, FcQAHead

READER_HEAD = {"LSTM": LstmQAHead, "CNN": CnnQAHead, "FC": FcQAHead}

class CustomModel(nn.Module):
    def __init__(self, backbone, head, pooling_pos, masking_ratio):
        super().__init__()
        self.backbone = backbone
        head_input_size = 768 # 현재 embedding 768 기준, 추후 argument로 수정 필요

        self.start_head = READER_HEAD[head](input_size=head_input_size)
        self.end_head = READER_HEAD[head](input_size=head_input_size)

        self.pooling_pos = pooling_pos
        self.masking_ratio = masking_ratio

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
            self.random_masking(input_ids) if self.training else input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]
        start_logits = self.start_head(sequence_output)
        end_logits = self.end_head(sequence_output)

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
            masking_ratio = args.train.masking_ratio
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
        )

        return trainer
