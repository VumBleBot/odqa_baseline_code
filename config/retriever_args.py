from typing import Optional
from dataclasses import dataclass, field


@dataclass
class RetrievalTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dense_train_dataset: Optional[str] = field(
        default="train_dataset", metadata={"help": "The name of the dataset to use."}
    )
    topk: Optional[int] = field(default=3)
    retrain: Optional[bool] = field(default=False, metadata={"help": "retriever 임베딩 재학습 인자"})
