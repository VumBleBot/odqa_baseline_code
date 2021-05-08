import wandb
import os.path as p
from itertools import product
from argparse import Namespace
from transformers import set_seed

from tools import update_args
from evaluation import evaluation
from prepare import get_dataset, get_reader
from slack_api import report_reader_to_slack


def train_reader(args):
    strategies = args.strategies
    seeds = args.seeds[: args.run_cnt]

    for idx, (seed, strategy) in enumerate(product(seeds, strategies)):
        wandb.init(project="p-stage-3", reinit=True)
        args = update_args(args, strategy)  # auto add args.save_path, args.base_path
        args.strategy, args.seed = strategy, seed
        args.info = Namespace()
        set_seed(seed)

        # below codes must run before 'reader.get_trainer()'
        # run_name: strategy + alias + seed
        args.train.run_name = "_".join([strategy, args.alias, str(seed)])
        args.train.output_dir = p.join(args.path.checkpoint, args.train.run_name)
        wandb.run.name = args.train.run_name

        print("checkpoint_dir: ", args.train.output_dir)

        datasets = get_dataset(args, is_train=True)
        reader = get_reader(args, datasets)

        # TODO: 아래 주석 아직도 유효한지 확인
        # retrieve 과정이 없어 top-k를 반환할 수 없음. 무조건 top-1만 반환
        reader.set_dataset(datasets, is_run=True)

        trainer = reader.get_trainer()

        if args.train.do_train:
            train_results = trainer.train()
            print(train_results)

        if args.train.do_eval:
            eval_results = trainer.evaluate()
            results = evaluation(args)
            eval_results["exact_match"] = results["EM"]["value"]
            eval_results["f1"] = results["F1"]["value"]

            print(eval_results)

            if args.report is True:
                report_reader_to_slack(args, p.basename(__file__), eval_results)


if __name__ == "__main__":
    from tools import get_args

    args = get_args()
    train_reader(args)
