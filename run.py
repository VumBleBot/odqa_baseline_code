import wandb
import os.path as p
from itertools import product
from argparse import Namespace
from transformers import set_seed

from tools import update_args
from prepare import prepare_dataset, get_reader_model, get_retriever


def train_reader(args):
    strategies = args.strategies
    seeds = args.seeds[: args.run_cnt]

    for idx, (seed, strategy) in enumerate(product(seeds, strategies)):
        wandb.init(project="p-stage-3", reinit=True)
        args = update_args(args, strategy)  # auto add args.save_path, args.base_path
        args.strategy, args.seed = strategy, seed
        args.info = Namespace()
        set_seed(seed)

        datasets = prepare_dataset(args, is_train=True)

        # below codes must run before 'reader.get_trainer()'
        # run_name: strategy + alias + seed
        args.train.run_name = "_".join([strategy, args.alias, str(seed)])
        args.train.output_dir = p.join(args.path.checkpoint, args.train.run_name)
        wandb.run.name = args.train.run_name

        print("checkpoint_dir: ", args.train.output_dir)

        reader = get_reader_model(args, datasets)
        retriever = get_retriever(args)

        datasets = retriever.retrieve_pipeline(args, datasets["validation"])
        reader.eval_dataset = reader.preprocess_dataset(datasets, is_train=False)

        trainer = reader.get_trainer()

        if args.train.do_train:
            train_results = trainer.train()
            print(train_results)

        if args.train.do_eval:
            eval_results = trainer.evaluate()
            print(eval_results)


if __name__ == "__main__":
    from tools import get_args

    args = get_args()
    train_reader(args)
