from typing import Optional
from dataclasses import dataclass, field


@dataclass
class RetrievalTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name_a: Optional[str] = field(default="train_dataset", metadata={"help": "The name of the dataset to use."})
    topk: Optional[int] = field(default=3)
