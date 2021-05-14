import json
import pprint
import os.path as p

import numpy as np
from fuzzywuzzy import fuzz

#  from datasets import Sequence, Value, Features, Dataset, concatenate_datasets

from datasets import concatenate_datasets
from tools import get_args
from prepare import get_dataset
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
            fancy_index.append(idx)

    return fancy_index


def make_ctx_dataset(args):
    args.model.tokenizer_name = "xlm-roberta-large"
    args.model.retriever_name = "BM25"

    bm25 = BM25Retrieval(args)
    bm25.get_embedding()

    train_dataset = get_train_dataset(args)

    with open(p.join(args.path.train_data_dir, "wikipedia_documents.json"), "r") as f:
        wiki_json = json.load(f)

    ctx_dataset_json = make_negative_ctx_dataset(bm25, train_dataset, wiki_json)

    save_path = p.join(args.path.train_data_dir, "negative_ctxs.json")

    with open(save_path, "w") as f:
        f.write(json.dumps(ctx_dataset_json, indent=4, ensure_ascii=False) + "\n")

    print(f"{save_path}에 저장되었습니다!")

    return ctx_dataset_json


def make_negative_ctx_dataset(bm25, train_dataset, wiki_json):
    ctx_dataset = []
    dataset_name = "klue-train-psgs_2100"

    scores, indices = bm25.get_relevant_doc_bulk(train_dataset["question"][:5], topk=50)
    scores, indices = np.array(scores), np.array(indices)
    bm25.contexts = np.array(bm25.contexts)

    for idx, question in enumerate(train_dataset["question"]):
        answers = train_dataset[idx]["answers"]["text"]  # EX) ['하원']
        text = train_dataset[idx]["context"]

        bm25_context_score = bm25.encoder.transform([text]).todense().sum()
        positive_ctxs = [
            {
                "text": text,  # EX) "미국.. 것이다."
                "title": train_dataset[idx]["title"],
                "score": bm25_context_score,  # bm25 점수
                "title_score": 0,
                "psg_id": train_dataset[idx]["document_id"],
            }
        ]

        negative_ctxs = []  # random으로
        hard_negative_ctxs = []  # bm25로 score 정렬

        # hard_negative_ctxs
        org_context = train_dataset[idx]["context"]

        # retrieved contexts
        ret_contexts = bm25.contexts[indices[idx]]
        fancy_index = delete_duplicate(org_context, ret_contexts)
        fancy_index = np.array(fancy_index)

        # duplicates removed retrieved contexts
        ret_scores = scores[idx][fancy_index]
        ret_doc_ids = indices[idx][fancy_index]
        ret_contexts = ret_contexts[fancy_index]

        for ret_score, ret_doc_id, ret_context in zip(ret_scores, ret_doc_ids, ret_contexts):
            ret_doc_id = str(ret_doc_id)
            tmp = {
                "title": wiki_json[ret_doc_id]["title"],
                "text": wiki_json[ret_doc_id]["text"],
                "score": ret_score,
                "title_score": 0,
                "psg_id": ret_doc_id,
            }

            hard_negative_ctxs.append(tmp)

        tmp = {
            "dataset": dataset_name,
            "question": question,
            "answers": answers,
            "positive_ctxs": positive_ctxs,
            "negative_ctxs": negative_ctxs,
            "hard_negative_ctxs": hard_negative_ctxs,
        }

        ctx_dataset.append(tmp)

        if idx == 0:
            pprint.pprint(ctx_dataset[0])
            break

    return ctx_dataset


def main(args):
    make_ctx_dataset(args)


if __name__ == "__main__":
    args = get_args()
    main(args)
