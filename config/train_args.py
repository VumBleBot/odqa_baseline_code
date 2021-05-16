from typing import Optional
from dataclasses import dataclass, field


@dataclass
class TrainArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    output_dir: Optional[str] = field(default="train_checkpoint", metadata={"help": "checkpoint_dir"})
    pororo_prediction: Optional[bool] = field(default=False, metadata={"help": "pororo mrc model voting(ensemble)"})
    do_ensemble: Optional[bool] = field(default=False, metadata={"help": "use ensemble.py for prediction"})
    # TrainArguments는 config/ST00.json 에서 수정해 주시면 됩니다.
