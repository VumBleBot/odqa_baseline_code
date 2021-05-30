from typing import Optional
from dataclasses import dataclass, field


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="bert-base-multilingual-cased",
        metadata={"help": "Reader Backbone, Path to pretrained model or model identifier from huggingface.co/models"},
    )
    model_path: str = field(default="", metadata={"help": "checkpoint path to load(used when evaluate&test)"})

    retriever_name: str = field(default="TFIDF", metadata={"help": "this args used in tools/get_retriever"})
    reader_name: str = field(default="DPR")

    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )

    tokenizer_name: Optional[str] = field(
        default="bert-base-multilingual-cased",
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
