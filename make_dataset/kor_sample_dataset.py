import os.path as p
from collections import defaultdict

import pandas as pd
from datasets import load_dataset
from datasets import concatenate_datasets
from datasets import Sequence, Value, Features, Dataset, DatasetDict

from tools import get_args


f = Features(
    {
        "answers": Sequence(
            feature={"text": Value(dtype="string", id=None), "answer_start": Value(dtype="int32", id=None)},
            length=-1,
            id=None,
        ),
        "context": Value(dtype="string", id=None),
        "id": Value(dtype="string", id=None),
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


def sampling_by_freq_weights(kor_dataset, weight="ans_start"):
    kor_df = kor_dataset.to_pandas()
    kor_ans_cnt = defaultdict(int)
    kor_ans_weights = defaultdict(float)
    bucket = 100

    for i, rows in kor_df.iterrows():
        kor_ans_cnt[rows["answers"]["answer_start"][0] // bucket] += 1

    total_cnt = sum(kor_ans_cnt.values())

    for k, v in kor_ans_cnt.items():
        kor_ans_weights[k] = (1 - (v / total_cnt)) ** 6  # 5가 적당

    def apply_weights(row):
        key = row["answer_start"][0] // bucket
        return kor_ans_weights[key]

    kor_df["weight"] = kor_df["answers"].apply(apply_weights)
    kor_df = kor_df.sample(n=3952 * 2, weights="weight", random_state=42)  # 다시 생각해보니깐 전체 저장은 불가능, 2배수 가능

    datasets = Dataset.from_pandas(kor_df, features=f)
    return datasets


def main(args):
    kor_dataset_path = p.join(args.path.train_data_dir, "kor_dataset")

    if p.exists(kor_dataset_path):
        raise FileExistsError("이미 존재하는 파일입니다!")

    kor_dataset = load_dataset("squad_kor_v1")

    kor_dataset = concatenate_datasets(
        [kor_dataset["train"].flatten_indices(), kor_dataset["validation"].flatten_indices()]
    )

    # (1) 문서 길이: KLUE MRC 512가 최소 길이
    kor_dataset = filtering_by_doc_len(kor_dataset, doc_len=512)

    # (2) 중복 Context 제거: Context당 최대 4개의 질문
    kor_dataset = filtering_by_dup_question(kor_dataset, dup_limit=4)

    # (3) KOR Weight Sampling 2배수 사용
    kor_dataset = sampling_by_freq_weights(kor_dataset)

    # (4) KOR_DATASET만 저장
    kor_datasets = DatasetDict({"train": kor_dataset})
    kor_datasets.save_to_disk(kor_dataset_path)

    print(f"{kor_dataset_path}에 저장되었습니다!")


if __name__ == "__main__":
    args = get_args()
    main(args)
