import torch.nn as nn
from transformers import AutoModel, AutoConfig, BertConfig, BertModel, AutoTokenizer, BertTokenizer, BertPreTrainedModel

from retrieval.dense import DprRetrieval
from tokenization_kobert import KoBertTokenizer


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


class BertEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]  # embedding 가져오기
        return pooled_output


class AutoEncoder(nn.Module):
    def __init__(self, model_name, config):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name, config=config)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.backbone(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]  # embedding 가져오기
        return pooled_output


class KoelectraEncoder(nn.Module):
    def __init__(self, model_name, config):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name, config=config)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.backbone(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[0][:, 0]  # 0번째 값이 [CLS] Token
        return pooled_output


class DprBert(DprRetrieval):
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
