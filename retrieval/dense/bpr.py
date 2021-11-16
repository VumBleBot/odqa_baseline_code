import torch
from argparse import Namespace
from transformers import BertConfig, BertModel, BertTokenizer, BertPreTrainedModel

from retrieval.dense import BprRetrieval

# class BertEncoder(BertPreTrainedModel):
#     def __init__(self, config):
#         super(BertEncoder, self).__init__(config)

#         self.bert = BertModel(config)
#         self.init_weights()
    
#     def forward(self, input_ids, attention_mask=None, token_type_ids=None):
#         outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
#         pooled_output = outputs[1]  # embedding 가져오기
#         return pooled_output

class BiEncoderModel(BertPreTrainedModel):
    # TODO: Check the type of args
    def __init__(self, config: BertConfig, args: Namespace ) -> None:
        super().__init__(config)

        self.args = args
        if getattr(args, "projection_dim_size", None) is not None:
            # TODO: Add "projection_dim_size" in args
            self.dense = torch.nn.Linear(config.hidden_size, args.projection_dim_size)
            self.layer_norm = torch.nn.LayerNorm(args.projection_dim_size, eps=config.layer_norm_eps)

        self.init_weights()

    # NOTE: Check if there's no problem in args?
    def forward(self, *args, **kwargs) -> torch.Tensor:
        sequence_output = super().forward(*args, **kwargs)[0]
        cls_output = sequence_output[:, 0, :].contiguous() # NOTE: look again 
        if getattr(self.args, "projection_dim_size", None) is not None:
            cls_output = self.layer_norm(self.dense(cls_output))

        return cls_output

class BiEncoder(BprRetrieval):
    def __init__(self) -> None:
        pass

    def convert_to_binary_code(self) -> None:
        pass
    
    # def loss  

# class BiEncoderModel(BertModel):
#     def __init__(self, config: BertConfig, hparams: Namespace):
#         super().__init__(config)

#         self.hparams = hparams
#         if getattr(hparams, "projection_dim_size", None) is not None:
#             self.dense = torch.nn.Linear(config.hidden_size, hparams.projection_dim_size)
#             self.layer_norm = torch.nn.LayerNorm(hparams.projection_dim_size, eps=config.layer_norm_eps)

#         self.init_weights()

#     def forward(self, *args, **kwargs) -> torch.Tensor:
#         sequence_output = super().forward(*args, **kwargs)[0]
#         cls_output = sequence_output[:, 0, :].contiguous() # NOTE: look again 
#         if getattr(self.hparams, "projection_dim_size", None) is not None:
#             cls_output = self.layer_norm(self.dense(cls_output))

#         return cls_output
