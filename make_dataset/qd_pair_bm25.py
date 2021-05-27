import os
import numpy as np
import pandas as pd
from datasets import Sequence, Value, Features, Dataset, concatenate_datasets

from utils.tools import get_args
from utils.prepare import get_dataset
from retrieval.sparse import BM25Retrieval


def get_questions(args):
    args.data.dataset_name = "train_dataset"
    datasets = get_dataset(args, is_train=True)
    datasets = concatenate_datasets([datasets["train"], datasets["validation"]])
    return datasets["question"], datasets["context"]


def make_dataset(args):
    args.model.tokenizer_name = "xlm-roberta-large"
    args.model.retriever_name = "BM25"

    bm25 = BM25Retrieval(args)
    bm25.get_embedding()

    # contexts: wiki
    documents = bm25.contexts
    questions, answers = get_questions(args)

    # question, contexts 형태의 데이터 셋을 생성
    make_negative_dataset(args, bm25, questions, answers, documents, "bm25_question_documents", num=16)

    # contexts: question
    bm25.contexts = questions
    bm25._exec_embedding  # Not Save, Just fit_transform

    # context, questions 형태의 데이터 셋을 생성
    make_negative_dataset(args, bm25, answers, questions, questions, "bm25_document_questions", num=32)


def make_negative_dataset(args, bm25, queries, answers, contexts, name, num=16):
    total = []
    scores, indices = bm25.get_relevant_doc_bulk(queries, topk=num * 2)

    answers, indices = np.array(answers, dtype="object"), np.array(indices)
    contexts = np.array(contexts, dtype="object")

    for idx, query in enumerate(queries):
        label = idx % num

        answer = answers[idx]
        context_list = contexts[indices[idx]]

        check_in = np.argwhere(context_list == answer)

        if check_in.shape[0] == 0:
            context_list[label] = answer
            context_list = context_list[:num]
        else:
            context_list[check_in[0][0]] = context_list[num]
            context_list[label] = answer
            context_list = context_list[:num]

        if idx % 100 == 0:
            print("query: ", query)
            print("answer: ", answer)
            print("negative:", context_list)
            print("label:", label)

        tmp = {"query": query, "negative_samples": context_list, "label": label}

        total.append(tmp)

    df = pd.DataFrame(total)

    f = Features(
        {
            "query": Value(dtype="string", id=None),
            "negative_samples": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
            "label": Value(dtype="int32", id=None),
        }
    )

    dataset = Dataset.from_pandas(df, features=f)
    dataset.save_to_disk(os.path.join(args.path.train_data_dir, name))


if __name__ == "__main__":
    args = get_args()
    make_dataset(args)
