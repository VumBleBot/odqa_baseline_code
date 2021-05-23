from transformers import ElectraConfig, ElectraModel, ElectraTokenizer, ElectraPreTrainedModel

from retrieval.dense import DprRetrieval


class ElectraEncoder(ElectraPreTrainedModel):
    def __init__(self, config):
        super(ElectraEncoder, self).__init__(config)

        self.electra = ElectraModel(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.electra.forward(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[0][:, 0]  # embedding 가져오기
        return pooled_output


class DprElectra(DprRetrieval):
    def __init__(self, args):
        super().__init__(args)
        self.backbone = "monologg/koelectra-base-v3-finetuned-korquad"
        self.tokenizer = ElectraTokenizer.from_pretrained(self.backbone)

    def _load_model(self):
        config = ElectraConfig.from_pretrained(self.backbone)
        p_encoder = ElectraEncoder.from_pretrained(self.backbone, config=config).cuda()
        q_encoder = ElectraEncoder.from_pretrained(self.backbone, config=config).cuda()
        return p_encoder, q_encoder

    def _get_encoder(self):
        config = ElectraConfig.from_pretrained(self.backbone)
        q_encoder = ElectraEncoder(config=config).cuda()
        return q_encoder
