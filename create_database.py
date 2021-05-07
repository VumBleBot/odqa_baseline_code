
from datasets import load_from_disk

import os
import tqdm
import pickle
import numpy as np
import pandas as pd
import os.path as p
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
import json
from datasets import Sequence, Value, Features, DatasetDict, Dataset

datasets = load_from_disk(p.join('/opt/ml/input/data/', 'train_dataset'))


class BM25Retrieval:
    def __init__(self):
        self.name = 'BM25'
        save_dir = p.join('/opt/ml/input/embed', self.name)
        if not p.exists(save_dir):
            os.mkdir(save_dir)

        self.embed_path = p.join(save_dir, "embedding.bin")
        self.encoder_path = p.join(save_dir, f"{self.name}.bin")

        self.encoder = None
        self.p_embedding = None

        with open(os.path.join('/opt/ml/input/', "data", "wikipedia_documents.json"), "r") as f:
            wiki = json.load(f)

        self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))

        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large", use_fast=True).tokenize

        self.b = 0.01
        self.k1 = 0.1
        self.encoder = TfidfVectorizer(tokenizer=self.tokenizer, ngram_range=(1, 2))

        self.avdl = None
        self.p_embedding = None

    def get_embedding(self):
        if p.isfile(self.embed_path) and p.isfile(self.encoder_path):
            with open(self.embed_path, "rb") as f:
                self.p_embedding = pickle.load(f)

            with open(self.encoder_path, "rb") as f:
                self.encoder = pickle.load(f)
        else:
            self.p_embedding, self.encoder = self._exec_embedding()

            with open(self.embed_path, "wb") as f:
                pickle.dump(self.p_embedding, f)

            with open(self.encoder_path, "wb") as f:
                pickle.dump(self.encoder, f)

        self.avdl = self.p_embedding.sum(1).mean()

    def _exec_embedding(self):
        self.encoder.fit(self.contexts)
        self.p_embedding = self.encoder.transform(self.contexts)
        return self.p_embedding, self.encoder

    def get_relevant_doc_bulk(self, queries, k=1):
        query_vecs = self.encoder.transform(queries)

        b, k1, avdl = self.b, self.k1, self.avdl
        len_p = self.p_embedding.sum(1).A1

        doc_scores = []
        doc_indices = []

        p_embedding = self.p_embedding.tocsc()

        for query_vec in tqdm.tqdm(query_vecs):
            p_emb_for_q = p_embedding.tocsc()[:, query_vec.indices]
            denom = p_emb_for_q + (k1 * (1 - b + b * len_p / avdl))[:, None]

            # idf(t) = log [ n / df(t) ] + 1 in sklearn, so it need to be converted
            # to idf(t) = log [ n / df(t) ] with minus 1
            idf = self.encoder._tfidf.idf_[None, query_vec.indices] - 1.0
            numer = p_emb_for_q.multiply(np.broadcast_to(idf, p_emb_for_q.shape)) * (k1 + 1)

            result = (numer / denom).sum(1).A1

            if not isinstance(result, np.ndarray):
                result = result.toarray()

            sorted_result_idx = np.argsort(result)[::-1]
            doc_score, doc_indice = result[sorted_result_idx].tolist()[:k], sorted_result_idx.tolist()[:k]
            doc_scores.append(doc_score)
            doc_indices.append(doc_indice)

        return doc_scores, doc_indices

    def retrieve(self, query_or_dataset, topk=1):
        assert self.p_embedding is not None, "get_embedding()을 먼저 수행한 후에 retrieve()를 작동시켜 주세요. "

        total = []
        doc_scores, doc_indices = self.get_relevant_doc_bulk(query_or_dataset["question"], k=topk)

        for idx, example in enumerate(query_or_dataset):
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
                    tmp["retrieval_label"] = 1 if tmp["context_id"] == example["document_id"] else 0
                total.append(tmp)

        df = pd.DataFrame(total)

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
                "original_context": Value(dtype="string", id=None),
                "retrieval_label": Value(dtype="int64", id=None)
            }
        )

        datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
        return datasets


retriever = BM25Retrieval()
retriever.get_embedding()
retrieved_dataset = retriever.retrieve(datasets["train"], topk=10)
retrieved_dataset["validation"].save_to_disk("/opt/ml/input/data/bm25_augmented_dataset")
