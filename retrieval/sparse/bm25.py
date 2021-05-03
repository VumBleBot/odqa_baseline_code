from tqdm.auto import tqdm
import pandas as pd
import numpy as np

from datasets import Dataset
from retrieval.sparse import SparseRetrieval


class BM25(SparseRetrieval):
    def __init__(self, args, tokenize_fn, context_path="wikipedia_documents.json", b=0.75, k1=1.6):
        super().__init__(args, tokenize_fn=tokenize_fn, context_path=context_path)
        self.name = "BM25"
        self.b = b
        self.k1 = k1
        self.avdl = None

    def get_sparse_embedding(self):
        super().get_sparse_embedding()
        self.avdl = self.p_embedding.sum(1).mean()

    def retrieve(self, query_or_dataset, topk=1):
        assert (
            self.p_embedding is not None
        ), "You must build faiss by self.get_sparse_embedding() before you run self.retrieve()."

        if isinstance(query_or_dataset, str):
            doc_scores = self.bm25.get_scores(query_or_dataset)
            topk_docs = self.bm25.get_top_n(query_or_dataset, self.context, topk)
            return doc_scores, topk_docs

        elif isinstance(query_or_dataset, Dataset):
            doc_scores = self.bm25.get_batch_scores(query_or_dataset, self.ids)
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

    def get_relevant_doc(self, query, k=1):

        query_vec = self.tfidfv.transform([query])
        assert np.sum(query_vec) != 0, "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        b, k1, avdl = self.b, self.k1, self.avdl

        len_context = self.p_embedding.sum(1).A1

        X = self.p_embedding.tocsc()[:, query_vec.indices]
        denom = X + (k1 * (1 - b + b * len_context/ avdl))[:, None]
        idf = self.tfidf._tfidf.idf_[None, query_vec.indices] - 1.
        numer = X.multiply(np.broadcast_to(idf, X.shape)) * (k1 + 1)

        result = (numer / denom).sum(1).A1

        if not isinstance(result, np.ndarray):
            result = result.toarray()

        sorted_result_idx = np.argsort(result.squeeze())[::-1]  # 내적값 내림차순 정렬한 인덱스 (test상 2269번 docs가 가장 내적값이 높음)
        return result.squeeze()[sorted_result_idx].tolist()[:k], sorted_result_idx.tolist()[:k]

    def get_relevant_doc_bulk(self, queries, k=1):
        query_vec = self.tfidfv.transform(queries)
        assert np.sum(query_vec) != 0, "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        b, k1, avdl = self.b, self.k1, self.avdl

        len_context = self.p_embedding.sum(1).A1

        X = self.p_embedding.tocsc()[:, query_vec.indices]
        denom = X + (k1 * (1 - b + b * len_context/ avdl))[:, None]
        idf = self.tfidf._tfidf.idf_[None, query_vec.indices] - 1.
        numer = X.multiply(np.broadcast_to(idf, X.shape)) * (k1 + 1)

        result = (numer / denom).sum(1).A1

        doc_scores = []
        doc_indices = []

        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])  # 스코어 리스트
            doc_indices.append(sorted_result.tolist()[:k])  # 예측 index 리스트

        return doc_scores, doc_indices
