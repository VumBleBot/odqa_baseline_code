from retrieval.dense import DprRetrieval
from tokenization_kobert import KoBertTokenizer


class DprKobertRetrieval(DprRetrieval):
    def __init__(self, args):
        super().__init__(args)
        self.backbone = "monologg/kobert"
        self.tokenizer = KoBertTokenizer.from_pretrained(self.backbone)
