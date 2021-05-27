import pprint
import os.path as p
from collections import defaultdict

import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from datasets import concatenate_datasets
from datasets import Value, Features, Dataset

from utils.tools import get_args
from utils.prepare import get_dataset
from retrieval.sparse import BM25Retrieval


def get_train_dataset(args):
    args.data.dataset_name = "train_dataset"
    datasets = get_dataset(args, is_train=True)
    datasets = concatenate_datasets([datasets["train"], datasets["validation"]])
    return datasets


def delete_duplicate(org_context, contexts):
    fancy_index = []

    for idx, context in enumerate(contexts):
        if fuzz.ratio(org_context, context) > 65:
            continue
        fancy_index.append(idx)

    return fancy_index


def make_triplet_dataset(bm25, train_dataset):
    triplet_datasets = defaultdict(list)

    scores, indices = bm25.get_relevant_doc_bulk(train_dataset["question"], topk=10)
    scores, indices = np.array(scores), np.array(indices)
    bm25.contexts = np.array(bm25.contexts)

    for idx, question in enumerate(train_dataset["question"]):
        triplet_datasets["question"].append(question)
        triplet_datasets["context"].append(train_dataset[idx]["context"])

        org_context = train_dataset[idx]["context"]

        # retrieved contexts
        ret_contexts = bm25.contexts[indices[idx]]
        fancy_index = delete_duplicate(org_context, ret_contexts)
        fancy_index = np.array(fancy_index)

        ret_contexts = ret_contexts[fancy_index]

        assert len(ret_contexts) != 0, "{ret_contexts} 중복으로 다 제거되었습다...."

        triplet_datasets["negative"].append(ret_contexts[0])

        if idx == 0:
            pprint.pprint(triplet_datasets)

    return triplet_datasets


def main(args):
    triplet_save_path = p.join(args.path.train_data_dir, "triplet_dataset")

    args.model.tokenizer_name = "xlm-roberta-large"
    args.model.retriever_name = "BM25"

    bm25 = BM25Retrieval(args)
    bm25.get_embedding()

    train_dataset = get_train_dataset(args)
    triplet_datasets = make_triplet_dataset(bm25, train_dataset)

    df = pd.DataFrame(triplet_datasets)

    f = Features(
        {
            "question": Value(dtype="string", id=None),
            "context": Value(dtype="string", id=None),
            "negative": Value(dtype="string", id=None),
        }
    )

    triplet_datasets = Dataset.from_pandas(df, features=f)
    triplet_datasets.save_to_disk(triplet_save_path)


if __name__ == "__main__":
    args = get_args()
    main(args)
