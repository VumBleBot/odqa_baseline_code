import torch, math
from typing import Dict, Tuple
from argparse import Namespace
from transformers import BertConfig, BertModel, BertTokenizer, BertPreTrainedModel

from retrieval.dense import BprRetrieval

class BiEncoderModel(BertPreTrainedModel):
    def __init__(self, config: BertConfig, args: Namespace) -> None:
        super().__init__(config)

        self.args = args
        self.bert = BertModel(config)
        self.training = False

        if getattr(args, "projection_dim_size", None) is not None:
            self.dense = torch.nn.Linear(config.hidden_size, args.retriever.projection_dim_size)
            self.layer_norm = torch.nn.LayerNorm(args.retriever.projection_dim_size, eps=config.layer_norm_eps)

        self.init_weights()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        sequence_output = self.bert(*args, **kwargs)[0] # NOTE: Need to dim check
        cls_output = sequence_output[:, 0, :].contiguous() # NOTE: look again
        if getattr(self.args, "projection_dim_size", None) is not None:
            cls_output = self.layer_norm(self.dense(cls_output))

        return cls_output

    def convert_to_binary_code(self, input_repr: torch.Tensor, global_step: int=1) -> torch.Tensor:
        if self.training:
            # NOTE : study this difference
            if self.args.retriever.use_ste:
                hard_input_repr = input_repr.new_ones(input_repr.size()).masked_fill_(input_repr < 0, -1.0)
                input_repr = torch.tanh(input_repr)
                return hard_input_repr + input_repr - input_repr.detach()
            else:
                scale = math.pow((1.0 + global_step * self.args.retriever.hashnet_gamma), 0.5)
                return torch.tanh(input_repr * scale)
        else:
            return input_repr.new_ones(input_repr.size()).masked_fill_(input_repr < 0, -1.0)

class BiEncoder(BprRetrieval):
    def __init__(self, args) -> None:
        super().__init__(args)

        self.args = args

        self.backbone = "bert-base-multilingual-cased"
        self.tokenizer = BertTokenizer.from_pretrained(self.backbone)

    def _load_model(self) -> BiEncoderModel:
        config = BertConfig.from_pretrained(self.backbone)
        p_encoder = BiEncoderModel.from_pretrained(self.backbone, config=config, args=self.args).cuda()
        q_encoder = BiEncoderModel.from_pretrained(self.backbone, config=config, args=self.args).cuda()
        return p_encoder, q_encoder

    # Get new encoder to load pretrained encoder
    def _get_encoder(self) -> BiEncoderModel:
        config = BertConfig.from_pretrained(self.backbone)
        q_encoder = BiEncoderModel(config=config).cuda()
        return q_encoder
