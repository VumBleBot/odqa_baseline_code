import os
import os.path as p
from glob import glob

from transformers.trainer_utils import get_last_checkpoint
from utils.tools import update_args, get_args
from utils.prepare import get_dataset, get_reader, get_retriever


def predict(args):
    # Don't use wandb

    strategies = args.strategies

    for idx, strategy in enumerate(strategies):
        args = update_args(args, strategy)
        args.strategy = strategy

        checkpoint_dir = glob(p.join(args.path.checkpoint, f"{strategy}*"))
        if not checkpoint_dir:
            raise FileNotFoundError(f"{strategy} 전략에 대한 checkpoint가 존재하지 않습니다.")

        args.model.model_path = get_last_checkpoint(checkpoint_dir[0])
        if args.model.model_path is None:
            raise FileNotFoundError(f"{checkpoint_dir[0]} 경로에 체크포인트가 존재하지 않습니다.")

        args.train.output_dir = p.join(args.path.checkpoint, strategy)
        args.train.do_predict = True

        datasets = get_dataset(args, is_train=False)
        reader = get_reader(args, eval_answers=datasets["validation"])
        retriever = get_retriever(args)

        datasets["validation"] = retriever.retrieve(datasets["validation"], topk=args.retriever.topk)["validation"]
        reader.set_dataset(eval_dataset=datasets["validation"])

        trainer = reader.get_trainer()

        # use pororo_predict WHERE args.train.pororo_predictions=True
        trainer.predict(test_dataset=reader.eval_dataset, test_examples=datasets["validation"])


if __name__ == "__main__":
    args = get_args()

    predict(args)
    print("Prediction finished.")
