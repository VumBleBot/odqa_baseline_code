import os
import json

import pandas as pd
from tqdm.auto import tqdm
from datasets import Dataset
from datasets import Sequence, Value, Features, DatasetDict

from make_dataset.aggregate_wiki import aggregate_wiki


class Retrieval:
    def __init__(self, args):
        self.args = args

        self.encoder = None
        self.p_embedding = None

        if args.data.wiki_agg:
            if not os.path.isfile(os.path.join(self.args.data_path, "data", "wikipedia_documents_agg.json")):
                aggregate_wiki(args)
            with open(os.path.join(self.args.data_path, "data", "wikipedia_documents_agg.json"), "r") as f:
                wiki = json.load(f)

        else:
            with open(os.path.join(self.args.data_path, "data", "wikipedia_documents.json"), "r") as f:
                wiki = json.load(f)

        self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))

    def _exec_embedding(self):
        raise NotImplementedError

    def get_embedding(self):
        raise NotImplementedError

    def get_relevant_doc_bulk(self, queries, k=1):
        raise NotImplementedError

    def retrieve(self, query_or_dataset, topk=1):
        assert self.p_embedding is not None, "get_embedding()을 먼저 수행한 후에 retrieve()를 작동시켜 주세요. "

        total = []
        doc_scores, doc_indices = self.get_relevant_doc_bulk(query_or_dataset["question"], k=topk)

        for idx, example in enumerate(tqdm(query_or_dataset, desc="Retrieval: ")):

            for doc_id in range(topk):
                tmp = {
                    "question": example["question"],
                    "id": example["id"],
                    "context_id": doc_indices[idx][doc_id],  # retrieved id
                    "context": self.contexts[doc_indices[idx][doc_id]],  # retrieved document
                }
                if "context" in example.keys() and "answers" in example.keys():
                    tmp["original_context"] = example["context"]  # original document
                    tmp["answers"] = example["answers"]  # original answer
                total.append(tmp)

        df = pd.DataFrame(total)

        if self.args.train.do_predict is True:
            f = Features(
                {
                    "context_id" : Value(dtype="int32", id=None),
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
                    "context_id": Value(dtype="int32", id=None),
                    "context": Value(dtype="string", id=None),
                    "id": Value(dtype="string", id=None),
                    "question": Value(dtype="string", id=None),
                    "original_context": Value(dtype="string", id=None),
                }
            )

        datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
        return datasets
