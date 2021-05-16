import wandb
import os.path as p
from itertools import product
from argparse import Namespace
from transformers import set_seed

from tools import update_args
from slack_api import report_reader_to_slack
from prepare import get_dataset, get_reader, get_retriever
from evaluation import evaluation


def train_reader(args):
    strategies = args.strategies
    seeds = args.seeds[: args.run_cnt]

    for idx, (seed, strategy) in enumerate(product(seeds, strategies)):
        wandb.init(project="p-stage-3", reinit=True)
        args = update_args(args, strategy)  # auto add args.save_path, args.base_path
        args.strategy, args.seed = strategy, seed
        args.info = Namespace()
        set_seed(seed)

        args.model.model_name_or_path = args.model_path

        # below codes must run before 'reader.get_trainer()'
        # run_name: strategy + alias + seed
        args.train.run_name = "_".join([strategy, args.alias, str(seed)])
        args.train.output_dir = p.join(args.path.checkpoint, args.train.run_name)
        wandb.run.name = args.train.run_name

        print("checkpoint_dir: ", args.train.output_dir)

        datasets = get_dataset(args, is_train=True)
        reader = get_reader(args, eval_answers=datasets["validation"])
        retriever = get_retriever(args)

        datasets["validation"] = retriever.retrieve(datasets["validation"], topk=args.retriever.topk)["validation"]
        reader.set_dataset(eval_dataset=datasets["validation"])

        trainer = reader.get_trainer()

        if args.train.do_eval:
            if args.train.pororo_prediction:
                eval_results, pororo_eval_results = trainer.evaluate()
                results, pororo_results = evaluation(args), evaluation(args, prefix="pororo_")

                for res, eval_res in zip((results, pororo_results), (eval_results, pororo_eval_results)):
                    eval_res["exact_match"] = res["EM"]["value"]
                    eval_res["f1"] = res["F1"]["value"]

                print("EVAL RESULT")
                print(eval_results)
                print("PORORO_EVAL RESULT")
                print(pororo_eval_results)

            else:
                eval_results = trainer.evaluate()
                eval_results = eval_results[0]
                results = evaluation(args)
                eval_results["exact_match"] = results["EM"]["value"]
                eval_results["f1"] = results["F1"]["value"]

                print("EVAL RESULT")
                print(eval_results)

            if args.report is True:
                report_reader_to_slack(args, p.basename(__file__), eval_results, use_pororo=False)
                if args.train.pororo_prediction is True:
                    report_reader_to_slack(args, p.basename(__file__), pororo_eval_results, use_pororo=True)


if __name__ == "__main__":
    from tools import get_args

    args = get_args()
    train_reader(args)
