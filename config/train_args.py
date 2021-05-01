from typing import Optional
from dataclasses import dataclass, field


@dataclass
class TrainArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    output_dir: Optional[str] = field(default="train_checkpoint", metadata={"help": "checkpoint_dir"})
    report_to: Optional[str] = field(default="wandb", metadata={"help": "wandb or tensorboard"})
    logging_steps: Optional[int] = field(default=100, metadata={"help": "logging_steps"})
