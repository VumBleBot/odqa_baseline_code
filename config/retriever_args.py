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

    # Parameters for bm25
    b: Optional[float] = field(default=0.01, metadata={"help":"0일 수록 문서 길이의 중요도가 낮아진다. 일반적으로 0.75 사용, 우리 모델에서 최적 0.01로 나옴"})
    k1: Optional[float] = field(default=0.1, metadata={"help":"TF의 saturation을 결정하는 요소. 어떤 토큰이 한 번 더 등장했을 때 이전에 비해 점수를 얼마나 높여주어야 하는가를 결정. (1.2~2.0을 사용하는 것이 일반적)"})
