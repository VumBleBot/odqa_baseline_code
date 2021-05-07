import os.path as p

from tools import update_args
from prepare import get_dataset, get_reader, get_retriever


def predict(args):
    # Don't use wandb

    strategies = args.strategies

    for idx, strategy in enumerate(strategies):
        args = update_args(args, strategy)  # auto add args.save_path, args.base_path
        args.strategy = strategy
        args.train.output_dir = p.join(args.path.checkpoint, strategy)

        datasets = get_dataset(args, is_train=False)
        retriever = get_retriever(args)
        reader = get_reader(args, datasets)

        datasets = retriever.retrieve(datasets["validation"], topk=args.retriever.topk)
        reader.set_dataset(datasets, is_run=False)

        trainer = reader.get_trainer()

        trainer.predict(test_dataset=reader.eval_dataset, test_examples=datasets["validation"])


if __name__ == "__main__":
    from tools import get_args

    args = get_args()
    args.train.do_predict = True

    predict(args)
