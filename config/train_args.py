from typing import Optional
from dataclasses import dataclass, field


@dataclass
class TrainArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    masking_ratio: Optional[float] = field(default=0.0, metadata={"help": "masking ratio"})
    do_eval_during_training: Optional[bool] = field(
        default=True, metadata={"help": "whether do evaluation when training"}
    )
    eval_step: Optional[int] = field(
        default=500,
        metadata={
            "help": "do evaluate every eval_step. If 'do_eval_during_training' is 'true', this argument will be ignored."
        },
    )

    pororo_prediction: Optional[bool] = field(default=False, metadata={"help": "pororo mrc model voting(ensemble)"})
    do_ensemble: Optional[bool] = field(default=False, metadata={"help": "use ensemble.py for prediction"})
    freeze_backbone: Optional[bool] = field(
        default=False, metadata={"help": "whether to freeze backbone during training"}
    )
    # TrainArguments는 config/ST00.json 에서 수정해 주시면 됩니다.
