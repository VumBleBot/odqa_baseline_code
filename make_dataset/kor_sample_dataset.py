import json
import os.path as p
from collections import defaultdict

import pandas as pd
from datasets import load_dataset
from datasets import concatenate_datasets
from datasets import Sequence, Value, Features, Dataset, DatasetDict

from utils.tools import get_args


f = Features(
    {
        "answers": Sequence(
            feature={"text": Value(dtype="string", id=None), "answer_start": Value(dtype="int32", id=None)},
            length=-1,
            id=None,
        ),
        "id": Value(dtype="string", id=None),
        "context": Value(dtype="string", id=None),
        "question": Value(dtype="string", id=None),
        "title": Value(dtype="string", id=None),
    }
)


def remove_multiple_indexes(rlist, indexes):
    assert indexes == sorted(indexes, reverse=True)

    for index in indexes:
        del rlist[index]

    return rlist


def filtering_by_doc_len(kor_dataset, doc_len=512):
    indexes = []

    for idx, context in enumerate(kor_dataset["context"]):
        if len(context) < doc_len:
            indexes.append(idx)

    indexes.sort(reverse=True)

    tmp = {}

    for key in kor_dataset.features.keys():
        tmp[key] = remove_multiple_indexes(kor_dataset[key], indexes)

    df = pd.DataFrame(tmp)
    datasets = Dataset.from_pandas(df, features=f)
    return datasets


def filtering_by_dup_question(kor_dataset, dup_limit=4):
    indexes = []
    context_cnt = defaultdict(int)

    for idx, context in enumerate(kor_dataset["context"]):
        context_cnt[context] += 1

        if context_cnt[context] > dup_limit:
            indexes.append(idx)

    indexes.sort(reverse=True)

    tmp = {}

    for key in kor_dataset.features.keys():
        tmp[key] = remove_multiple_indexes(kor_dataset[key], indexes)

    df = pd.DataFrame(tmp)
    datasets = Dataset.from_pandas(df, features=f)
    return datasets


def sampling_by_ans_start_weights(kor_dataset, sample=8000):
    kor_df = kor_dataset.to_pandas()
    kor_ans_cnt = defaultdict(int)
    kor_ans_weights = defaultdict(float)
    bucket = 100

    for i, rows in kor_df.iterrows():
        kor_ans_cnt[rows["answers"]["answer_start"][0] // bucket] += 1

    total_cnt = sum(kor_ans_cnt.values())

    for k, v in kor_ans_cnt.items():
        kor_ans_weights[k] = (1 - (v / total_cnt)) ** 6  # 5??? ??????

    def apply_weights(row):
        key = row["answer_start"][0] // bucket
        return kor_ans_weights[key]

    kor_df["weight"] = kor_df["answers"].apply(apply_weights)
    kor_df = kor_df.sample(n=sample, weights="weight", random_state=42)  # ?????? ?????????????????? ?????? ????????? ?????????, 2?????? ??????

    datasets = Dataset.from_pandas(kor_df, features=f)
    return datasets


def sampling_by_doc_lens(kor_dataset, sample):
    kor_df = kor_dataset.to_pandas()
    kor_ans_cnt = defaultdict(int)
    kor_ans_weights = defaultdict(float)
    bucket = 100

    for i, rows in kor_df.iterrows():
        kor_ans_cnt[len(rows["context"]) // bucket] += 1

    total_cnt = sum(kor_ans_cnt.values())

    for k, v in kor_ans_cnt.items():
        kor_ans_weights[k] = (1 - (v / total_cnt)) ** 6  # 5??? ??????

    def apply_weights(row):
        key = len(row) // bucket
        return kor_ans_weights[key]

    kor_df["weight"] = kor_df["context"].apply(apply_weights)
    kor_df = kor_df.sample(n=sample, weights="weight", random_state=42)  # ?????? ?????????????????? ?????? ????????? ?????????, 2?????? ??????

    datasets = Dataset.from_pandas(kor_df, features=f)
    return datasets


def make_kor_dataset_v1(args):
    """KorQuad Dataset V1
    1. ?????? ?????? 512?????? Filtering
    2. Context??? Question ?????? 4???
    3. ans_start ????????? 8000??? ?????????
    """

    kor_dataset_path = p.join(args.path.train_data_dir, "kor_dataset")

    if p.exists(kor_dataset_path):
        raise FileExistsError(f"{kor_dataset_path}??? ?????? ???????????? ???????????????!")

    kor_dataset = load_dataset("squad_kor_v1")

    kor_dataset = concatenate_datasets(
        [kor_dataset["train"].flatten_indices(), kor_dataset["validation"].flatten_indices()]
    )

    # (1) ?????? ??????: KLUE MRC 512??? ?????? ??????
    kor_dataset = filtering_by_doc_len(kor_dataset, doc_len=512)

    # (2) ?????? Context ??????: Context??? ?????? 4?????? ??????
    kor_dataset = filtering_by_dup_question(kor_dataset, dup_limit=4)

    # (3) KOR answer_start Weight Sampling 2?????? ??????
    kor_dataset = sampling_by_ans_start_weights(kor_dataset, sample=8000)

    # (4) KOR_DATASET??? ??????
    kor_datasets = DatasetDict({"train": kor_dataset})
    kor_datasets.save_to_disk(kor_dataset_path)

    print(f"{kor_dataset_path}??? ?????????????????????!")


def make_kor_dataset_v2(args):
    """KorQuad Dataset V1
    1. ?????? ?????? 512?????? Filtering
    2. Context??? Question ?????? 4???
    3. ans_start ????????? 8000??? ?????????
    4. doc_len ?????????  4000??? ?????????
    """
    kor_dataset_path = p.join(args.path.train_data_dir, "kor_dataset_v2")

    if p.exists(kor_dataset_path):
        raise FileExistsError(f"{kor_dataset_path}??? ?????? ???????????? ???????????????!")

    kor_dataset = load_dataset("squad_kor_v1")

    kor_dataset = concatenate_datasets(
        [kor_dataset["train"].flatten_indices(), kor_dataset["validation"].flatten_indices()]
    )

    # (1) ?????? ??????: KLUE MRC 512??? ?????? ??????
    kor_dataset = filtering_by_doc_len(kor_dataset, doc_len=512)

    # (2) ?????? Context ??????: Context??? ?????? 4?????? ??????
    kor_dataset = filtering_by_dup_question(kor_dataset, dup_limit=4)

    # (3) KOR answer_start Weight Sampling 2?????? ??????
    kor_dataset = sampling_by_ans_start_weights(kor_dataset)

    # (4) KOR docs_len Weights Sampling 4000??? ??????
    kor_dataset = sampling_by_doc_lens(kor_dataset, sample=4000)

    # (5) KOR_DATASET??? ??????
    kor_datasets = DatasetDict({"train": kor_dataset})
    kor_datasets.save_to_disk(kor_dataset_path)

    print(f"{kor_dataset_path}??? ?????????????????????!")


def get_etr_dataset(args):
    etr_path = p.join(args.path.train_data_dir, "etr_qa_dataset.json")

    if not p.exists(etr_path):
        raise FileNotFoundError(f"ETRI ????????? ??? {etr_path}??? ????????? ????????? ????????? ??????????????? ????????????.")

    with open(etr_path, "r") as f:
        etr_dict = json.load(f)

    #  print(etr_dict["data"][0])
    new_dataset = defaultdict(list)

    cnt = 0

    for datas in etr_dict["data"]:
        title = datas["title"]
        context = datas["paragraphs"][0]["context"]

        for questions in datas["paragraphs"][0]["qas"]:
            question = questions["question"]
            answers = {
                "answer_start": [questions["answers"][0]["answer_start"]],
                "text": [questions["answers"][0]["text"]],
            }

            new_dataset["id"].append(f"etr-custom-{cnt}")
            new_dataset["title"].append(title)
            new_dataset["context"].append(context)
            new_dataset["question"].append(question)
            new_dataset["answers"].append(answers)

            cnt += 1

    f = Features(
        {
            "answers": Sequence(
                feature={"text": Value(dtype="string", id=None), "answer_start": Value(dtype="int32", id=None)},
                length=-1,
                id=None,
            ),
            "id": Value(dtype="string", id=None),
            "context": Value(dtype="string", id=None),
            "question": Value(dtype="string", id=None),
            "title": Value(dtype="string", id=None),
        }
    )

    df = pd.DataFrame(new_dataset)
    etr_dataset = Dataset.from_pandas(df, features=f)

    return etr_dataset


def make_etr_dataset_v1(args):
    """ETRI ????????? ??? ???????????? ??????
    1. ?????? ?????? 512?????? Filtering
    2. ?????? Context ??????, Question ?????? 4???
    3. ans_start ????????? 3000??? ?????????
    """
    etr_dataset_path = p.join(args.path.train_data_dir, "etr_dataset_v1")

    if p.exists(etr_dataset_path):
        raise FileExistsError(f"{etr_dataset_path}??? ?????? ???????????? ???????????????!")

    etr_dataset = get_etr_dataset(args)

    # (1) ?????? ??????: KLUE MRC 512??? ?????? ??????
    etr_dataset = filtering_by_doc_len(etr_dataset, doc_len=512)

    # (2) ?????? Context ??????: Context??? ?????? 4?????? ??????
    etr_dataset = filtering_by_dup_question(etr_dataset, dup_limit=4)

    # (3) ETR answer_start Weight 3000??? Sampling
    etr_dataset = sampling_by_ans_start_weights(etr_dataset, sample=3000)

    # (4) ETR_DATASET??? ??????
    etr_datasets = DatasetDict({"train": etr_dataset})
    etr_datasets.save_to_disk(etr_dataset_path)

    print(f"{etr_dataset_path}??? ?????????????????????!")


def main(args):
    make_kor_dataset_v1(args)
    make_kor_dataset_v2(args)
    make_etr_dataset_v1(args)


if __name__ == "__main__":
    args = get_args()
    main(args)
