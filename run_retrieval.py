import random
from itertools import product
from collections import defaultdict

import wandb
import matplotlib.pyplot as plt
from transformers import set_seed

from tools import update_args
from prepare import get_dataset, get_retriever


def get_topk_fig(args, topk_result):
    dup = set()

    colors = ["#ED5934", "#A3ED40", "#ED2867", "#11EDA4", "#C91CED"]
    markers = ["8", "*", "^", "D", "X", "+"]
    linestyles = ["solid", "dashed", "dashdot", "dotted"]

    def get_cml():
        tries = 10

        while tries:
            co = random.choice(colors)
            mk = random.choice(markers)
            li = random.choice(linestyles)

            if tuple((co, mk, li)) not in dup:
                return co, mk, li

            dup.add(tuple((co, mk, li)))
            tries -= 1

        raise TimeoutError("코드 이상한 듯")

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
    ax.set_title("Compare TOP-K", loc="left", fontsize=12, va="bottom", fontweight="semibold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim([0, 1])

    x_pos = range(1, args.retriever.topk + 1)
    ax.set_xticks(x_pos)

    for name, topk in topk_result.items():
        co, mk, li = get_cml()
        ax.plot(x_pos, topk, color=co, marker=mk, linestyle=li)
        ax.text(
            x_pos[-1] + 0.1,
            topk[-1],
            s=name,
            fontweight="bold",
            va="center",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="#F36E8E", ec="black", alpha=0.3),
        )

    return fig


def train_retriever(args):
    strategies = args.strategies
    seeds = args.seeds[: args.run_cnt]

    topk_result = defaultdict(list)
    wandb.init(project="p-stage-3", reinit=True)
    wandb.run.name = "COMPARE RETRIEVER"

    for idx, (seed, strategy) in enumerate(product(seeds, strategies)):
        args = update_args(args, strategy)  # auto add args.save_path, args.base_path
        set_seed(seed)

        datasets = get_dataset(args, is_train=True)
        origin_datasets = datasets["validation"]
        retriever = get_retriever(args)
        datasets = retriever.retrieve(datasets["validation"])

        id_to_context = {row["id"]: row["context"] for row in origin_datasets}
        cur_cnt, tot_cnt = 0, len(origin_datasets)

        print(f"전략: {strategy} RETRIEVER: {args.model.retriever_name}")
        legend_name = "_".join([strategy, args.model.retriever_name])

        for idx, topk_dataset in enumerate(zip([datasets["validation"][i::10] for i in range(args.retriever.topk)])):
            for row in zip(topk_dataset[0]["id"], topk_dataset[0]["context"]):
                if id_to_context[row[0]] == row[1]:
                    cur_cnt += 1

            topk_acc = cur_cnt / tot_cnt
            topk_result[legend_name].append(topk_acc)
            print(f"TOPK: {idx} ACC: {topk_acc * 100:.2f}")

    fig = get_topk_fig(args, topk_result)
    wandb.log({"retriever topk result": wandb.Image(fig)})


if __name__ == "__main__":
    from tools import get_args

    args = get_args()
    train_retriever(args)
