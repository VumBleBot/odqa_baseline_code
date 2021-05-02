import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import Sequence, Value, Features, DatasetDict

from tqdm.auto import tqdm
import pandas as pd
import pickle
import json
import os
import numpy as np

from datasets import Dataset


class SparseRetrieval:
    def __init__(self, args, tokenize_fn, context_path="wikipedia_documents.json"):
        self.args = args
        self.name = "TFIDF"

        with open(os.path.join(self.args.data_path, "data", context_path), "r") as f:
            wiki = json.load(f)

        self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))  # 데이터셋에는 56737개의 unique contexts 존재
        self.ids = list(range(len(self.contexts)))

        self.tfidfv = TfidfVectorizer(tokenizer=tokenize_fn, ngram_range=(1, 2), max_features=50000)

        self.p_embedding = None
        self.indexer = None

    def get_sparse_embedding(self):
        pickle_name = "embedding.bin"
        tfidfv_name = "tfidv.bin"

        embed_dir = os.path.join(self.args.path.embed, self.name)

        if not os.path.exists(embed_dir):
            os.mkdir(embed_dir)

        emd_path = os.path.join(embed_dir, pickle_name)
        tfidfv_path = os.path.join(embed_dir, tfidfv_name)

        if os.path.isfile(emd_path) and os.path.isfile(tfidfv_path):
            with open(emd_path, "rb") as file:
                self.p_embedding = pickle.load(file)
            with open(tfidfv_path, "rb") as file:
                self.tfidfv = pickle.load(file)
        else:
            self.p_embedding = self.tfidfv.fit_transform(self.contexts)
            with open(emd_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            with open(tfidfv_path, "wb") as file:
                pickle.dump(self.tfidfv, file)

    def build_faiss(self):
        num_clusters, niter = 16, 5

        p_emb = self.p_embedding.astype(np.float32).toarray()
        emb_dim = p_emb.shape[-1]
        index_flat = faiss.IndexFlatL2(emb_dim)  # 그래서 index_flat의 d = 50000.

        clus = faiss.Clustering(emb_dim, num_clusters)
        clus.verbose = True
        clus.niter = niter
        clus.train(p_emb, index_flat)

        centroids = faiss.vector_float_to_array(clus.centroids)
        centroids = centroids.reshape(num_clusters, emb_dim)  # flatten reshape

        quantizer = faiss.IndexFlatL2(emb_dim)
        quantizer.add(centroids)

        self.indexer = faiss.IndexIVFScalarQuantizer(quantizer, quantizer.d, quantizer.ntotal, faiss.METRIC_L2)
        self.indexer.train(p_emb)
        self.indexer.add(p_emb)

    def retrieve_pipeline(self, args, query_or_dataset, topk=1):
        """
        Create retrieved result dataframe as refined form. Then return it as DatasetDict.
        Before:
            - features: ['__index_level_0__', 'answers', 'context', 'document_id', 'id', 'question', 'title']
                - 'id' : question(query) id. (ex - mrc-0-0000,...)
                - 'answers'
                    - 'answer_start' : offset position number of start character in answer string.
                    - 'text' : ground-truth answer string. It is answer for question(query).
        After:
            - validation
                - features: ['answers', 'context', 'id', 'question']
            - predict : exclude 'answers' from validation features. The result doesn't need to be scored.
                - features: ['context', 'id', 'question']

        :param args
            - train.do_predict
        :param query_or_dataset: single string query or dataset to retrieve.
        :param topk: Number which retriever returns from retrieved result.
        :return: DatasetDict of retrieved result (as refined form).
        """
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

    def retrieve(self, query_or_dataset, topk=1):
        assert (
            self.p_embedding is not None
        ), "You must build faiss by self.get_sparse_embedding() before you run self.retrieve()."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            return doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)]

        elif isinstance(query_or_dataset, Dataset):
            total = []
            doc_scores, doc_indices = self.get_relevant_doc_bulk(query_or_dataset["question"], k=1)

            for idx, example in enumerate(tqdm(query_or_dataset, desc="Sparse retrieval: ")):
                tmp = {
                    "question": example["question"],
                    "id": example["id"],
                    "context_id": doc_indices[idx][0],  # retrieved document id
                    "context": self.contexts[doc_indices[idx][0]],  # retrieved document
                }
                if "context" in example.keys() and "answers" in example.keys():
                    tmp["original_context"] = example["context"]  # original document(ground-truth context)
                    tmp["answers"] = example["answers"]  # original answer(ground-truth answer)
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc(self, query, k=1):
        query_vec = self.tfidfv.transform([query])
        assert np.sum(query_vec) != 0, "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        result = query_vec * self.p_embedding.T

        if not isinstance(result, np.ndarray):
            result = result.toarray()

        sorted_result_idx = np.argsort(result.squeeze())[::-1]  # 내적값 내림차순 정렬한 인덱스 (test상 2269번 docs가 가장 내적값이 높음)
        return result.squeeze()[sorted_result_idx].tolist()[:k], sorted_result_idx.tolist()[:k]

    def get_relevant_doc_bulk(self, queries, k=1):
        query_vec = self.tfidfv.transform(queries)
        assert np.sum(query_vec) != 0, "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        result = query_vec * self.p_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()

        doc_scores = []
        doc_indices = []

        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])  # 스코어 리스트
            doc_indices.append(sorted_result.tolist()[:k])  # 예측 index 리스트

        return doc_scores, doc_indices

    def retrieve_faiss(self, query_or_dataset, topk=1):
        assert (
            self.indexer is not None
        ), "You must build faiss by self.build_faiss() before you run self.retrieve_faiss()."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc_faiss(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print("Top-%d passage with score %.4f" % (i + 1, doc_scores[i]))
                print(self.contexts[doc_indices[i]])
            return doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)]

        elif isinstance(query_or_dataset, Dataset):
            queries = query_or_dataset["question"]
            total = []

            doc_scores, doc_indices = self.get_relevant_doc_bulk_faiss(queries, k=topk)  # TODO 여기서 병목이 있는데...

            for idx, example in enumerate(tqdm(query_or_dataset, desc="Sparse retrieval: ")):
                tmp = {
                    "question": example["question"],
                    "id": example["id"],
                    "context_id": doc_indices[idx][0],
                    "context": self.contexts[doc_indices[idx][0]],
                }

                if "context" in example.keys() and "answers" in example.keys():
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]

                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc_faiss(self, query, k=1):
        query_vec = self.tfidfv.transform([query])
        assert np.sum(query_vec) != 0, "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        q_emb = query_vec.astype(np.float32).toarray()
        D, I = self.indexer.search(q_emb, k)
        return D.tolist()[0], I.tolist()[0]

    def get_relevant_doc_bulk_faiss(self, queries, k=1):
        query_vecs = self.tfidfv.transform(queries)
        assert np.sum(query_vecs) != 0, "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        q_embs = query_vecs.toarray().astype(np.float32)
        D, I = self.indexer.search(q_embs, k)
        return D.tolist(), I.tolist()
