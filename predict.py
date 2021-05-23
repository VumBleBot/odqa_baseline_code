import os.path as p

from utils.tools import update_args, get_args
from utils.prepare import get_dataset, get_reader, get_retriever


def predict(args):
    # Don't use wandb

    strategies = args.strategies

    for idx, strategy in enumerate(strategies):
        args = update_args(args, strategy)  # auto add args.save_path, args.base_path
        args.strategy = strategy

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
