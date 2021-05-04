import os
import json

import pandas as pd
from tqdm.auto import tqdm
from datasets import Dataset
from datasets import Sequence, Value, Features, DatasetDict


class Retrieval:
    def __init__(self, args):
        self.args = args

        self.encoder = None
        self.p_embedding = None

        with open(os.path.join(self.args.data_path, "data", "wikipedia_documents.json"), "r") as f:
            wiki = json.load(f)

        self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))  # 데이터셋에는 56737개의 unique contexts 존재

    def exec_embedding(self):  # tf-idf의 경우는 vecotrizer, 다른 애는
        raise NotImplementedError

    def get_embedding(self):
        raise NotImplementedError

    def get_relevant_doc_bulk(self, queries, k=1):
        raise NotImplementedError

    def retrieve_pipeline(self, args, query_or_dataset, topk=1):
        df = self.retrieve(query_or_dataset, topk=topk)

        if args.train.do_predict is True:
            f = Features(
                {
                    "context": Value(dtype="string", id=None),
                    "id": Value(dtype="string", id=None),
                    "question": Value(dtype="string", id=None),
                }
            )
        else:
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
                }
            )

        datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
        return datasets

    def retrieve(self, query_or_dataset, topk=1):  # buld query만을 가정 (query_or_dataset, Dataset, topk)
        assert (
            self.p_embedding is not None
        ), "You must build faiss by self.get_sparse_embedding() before you run self.retrieve()."  # (Question)

        total = []
        doc_scores, doc_indices = self.get_relevant_doc_bulk(query_or_dataset["question"], k=1)

        for idx, example in enumerate(tqdm(query_or_dataset, desc="Sparse retrieval: ")):
            tmp = {
                "question": example["question"],
                "id": example["id"],
                "context_id": doc_indices[idx][0],  # retrieved id
                "context": self.contexts[doc_indices[idx][0]],  # retrieved doument
            }
            if "context" in example.keys() and "answers" in example.keys():
                tmp["original_context"] = example["context"]  # original document
                tmp["answers"] = example["answers"]  # original answer
            total.append(tmp)

        cqas = pd.DataFrame(total)
        return cqas
