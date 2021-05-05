from transformers import BertTokenizer
from retrieval.dense import DprRetrieval


class DprKobertRetrieval(DprRetrieval):
    def __init__(self, args):
        super().__init__(args)
        self.backbone = "monologg/kobert"
        self.tokenizer = BertTokenizer.from_pretrained(self.backbone)
