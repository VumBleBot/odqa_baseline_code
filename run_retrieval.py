from itertools import product
from argparse import Namespace
from collections import defaultdict

import wandb
import matplotlib.pyplot as plt
from transformers import set_seed

from tools import update_args
from prepare import get_dataset, get_retriever


def get_topk_fig(topk_result):

    fig, ax = plt.subplots(1, 1)

    for k, v in topk_result.items():
        pass


def train_retriever(args):
    strategies = args.strategies
    seeds = args.seeds[: args.run_cnt]

    topk_result = defaultdict(list)

    for idx, (seed, strategy) in enumerate(product(seeds, strategies)):
        args = update_args(args, strategy)  # auto add args.save_path, args.base_path
        args.strategy, args.seed = strategy, seed
        args.info = Namespace()
        set_seed(seed)

        wandb.run.name = "_".join([strategy, args.alias, str(seed)]) + "_retriever"

        datasets = get_dataset(args, is_train=True)
        origin_datasets = datasets["validation"]

        retriever = get_retriever(args, topk=args.retriever.topk)
        datasets = retriever.retrieve(datasets["validation"])

        id_to_context = {row["id"]: row["context"] for row in origin_datasets}

        cur_cnt, tot_cnt = 0, len(origin_datasets)

        for topk_dataset in zip([datasets["validation"][i::10] for i in range(args.retriever.topk)]):
            for row in zip(topk_dataset[0]["id"], topk_dataset[0]["context"]):
                if id_to_context[row[0]] == row[1]:
                    cur_cnt += 1

            topk_result[wandb.run.name].append(cur_cnt / tot_cnt)

    fig = get_topk_fig(topk_result)


if __name__ == "__main__":
    from tools import get_args

    args = get_args()
    train_retriever(args)
