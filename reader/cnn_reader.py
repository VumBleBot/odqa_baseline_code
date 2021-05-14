import numpy as np
import torch
from torch import nn
from trainer_qa import QuestionAnsweringTrainer

from transformers.modeling_outputs import QuestionAnsweringModelOutput
from reader.base_reader import BaseReader

class CnnQAHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_1 = nn.Conv1d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)
        self.conv_3 = nn.Conv1d(in_channels=in_channels, out_channels=1, kernel_size=3, padding=1)
        self.conv_5 = nn.Conv1d(in_channels=in_channels, out_channels=1, kernel_size=5, padding=2)

    def forward(self, x):
        x = x.transpose(1, 2).contiguous()
        conv1_out = self.conv_1(x).transpose(1, 2).contiguous().squeeze(-1)
        conv3_out = self.conv_3(x).transpose(1, 2).contiguous().squeeze(-1)
        conv5_out = self.conv_5(x).transpose(1, 2).contiguous().squeeze(-1)
        x = conv1_out + conv3_out + conv5_out

        # return conv3_out
        return x
        
class CnnReaderModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        # BERT 모델의 경우 add_pooling_layer=False 옵션 추가 필요('bert-base-multilingual-uncased')
        self.backbone = backbone
        head_input_size = 768 # !!추후 수정 필요!! (자동화)
        # 현재 'monologg/koelectra-base-v3-finetuned-korquad' 기준

        self.start_head = CnnQAHead(in_channels=head_input_size)
        self.end_head = CnnQAHead(in_channels=head_input_size)
    
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
        discriminator_hidden_states = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = discriminator_hidden_states[0]
        start_logits = self.start_head(sequence_output)
        end_logits = self.end_head(sequence_output)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
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
            ) + discriminator_hidden_states[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )

class CnnHeadReader(BaseReader):
    def __init__(self, args, model, tokenizer, eval_answers):
        super().__init__(args, None, tokenizer, eval_answers)
        self.model = CnnReaderModel(backbone=model)
        if args.model.model_path:
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
