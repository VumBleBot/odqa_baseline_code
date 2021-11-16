from transformers import BertConfig, BertModel, BertTokenizer, BertPreTrainedModel

from retrieval.dense import DpRetrieval


class BertEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]  # embedding 가져오기
        return pooled_output


class DpBert(DpRetrieval):
    def __init__(self, args):
        super().__init__(args)
        self.backbone = "bert-base-multilingual-cased"
        self.tokenizer = BertTokenizer.from_pretrained(self.backbone)

    def _load_model(self):
        config = BertConfig.from_pretrained(self.backbone)
        p_encoder = BertEncoder.from_pretrained(self.backbone, config=config).cuda()
        q_encoder = BertEncoder.from_pretrained(self.backbone, config=config).cuda()
        return p_encoder, q_encoder

    def _get_encoder(self):
        config = BertConfig.from_pretrained(self.backbone)
        q_encoder = BertEncoder(config=config).cuda()
        return q_encoder
