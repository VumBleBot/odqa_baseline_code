import os
import os.path as p
import pickle
import tqdm 
import numpy as np
from retrieval.base_retrieval import Retrieval
from sklearn.linear_model import LogisticRegression
from scipy.special import softmax
from datasets import load_from_disk, concatenate_datasets

class HybridRetrieval(Retrieval):
    """ 이미 학습된 Sparse, Dense Retriever를 사용한다."""
    def __init__(self, args):
        super().__init__(args)
        self.sparse_retriever = None
        self.dense_retriever = None

    def get_embedding(self):
        self.sparse_retriever.get_embedding()
        self.dense_retriever.get_embedding()
        self.p_embedding = 1  # fake for super().retrieve's, assert line

    def _rank_fusion_by_hybrid(self, dense_hits, sparse_hits):
        ranks = []
        min_dense_score, min_sparse_score = min(dense_hits.values()), min(sparse_hits.values())

        for doc in set(dense_hits.keys()) | set(sparse_hits.keys()):
            if doc not in dense_hits:  # sparse 에만 들었을 때
                score = self.args.retriever.alpha * sparse_hits[doc] + min_dense_score
            elif doc not in sparse_hits:  # dense 에만 들었을 때
                score = self.args.retriever.alpha * min_sparse_score + dense_hits[doc]
            else:  # 둘 다 들었을 때
                score = self.args.retriever.alpha * sparse_hits[doc] + dense_hits[doc]

            ranks.append((doc, score))

        ranks = sorted(ranks, key=lambda x: x[1], reverse=True)[: self.args.retriever.topk]

        doc_index, doc_score = zip(*ranks)
        doc_index, doc_score = list(doc_index), list(doc_score)

        return doc_score, doc_index

    def get_relevant_doc_bulk(self, queries, topk):
        dense_scores, dense_indices = self.dense_retriever.get_relevant_doc_bulk(queries, topk)
        sparse_scores, sparse_indices = self.sparse_retriever.get_relevant_doc_bulk(queries, topk)

        doc_scores, doc_indices = [], []

        for idx, _ in enumerate(queries):
            d_scores, d_indices = dense_scores[idx], dense_indices[idx]
            dense_hits = {d_indices[k]: d_scores[k] for k in range(topk)}

            s_scores, s_indices = sparse_scores[idx], sparse_indices[idx]
            sparse_hits = {s_indices[k]: s_scores[k] for k in range(len(topk))}

            doc_score, doc_index = self._rank_fusion_by_hybrid(dense_hits, sparse_hits)

            doc_scores.append(doc_score)
            doc_indices.append(doc_index)

        return doc_scores, doc_indices


class HybridLogisticRetrieval(Retrieval):
    def __init__(self, args):
        super().__init__(args)
        self.sparse_retriever = None
        self.dense_retriever = None
        self.logistic = None 
        self.num_features = 3
        self.kbound = 3
        
    def get_embedding(self):
        self.sparse_retriever.get_embedding()
        self.dense_retriever.get_embedding()
        self._get_logistic_regression()
        self.p_embedding = 1  # fake for super().retrieve's, assert line

    def _exec_logistic_regression(self):
        datasets = load_from_disk(p.join(self.args.path.train_data_dir, 'train_dataset')) 
        
        train_dataset = concatenate_datasets([
            datasets['train'],
            datasets['validation']
        ])

        queries = train_dataset['question']
        doc_scores, doc_indices = self.sparse_retriever.get_relevant_doc_bulk(queries, topk=8)
        doc_scores, doc_indices = np.array(doc_scores), np.array(doc_indices)

        contexts = np.array(self.sparse_retriever.contexts)

        train_x, train_y = [], [] 
    

        for idx in tqdm.tqdm(range(len(doc_scores))):
            doc_index = doc_indices[idx]
            org_context = train_dataset['context'][idx]
            
            feature_vector = [doc_scores[idx][:pow(2, i)] for i in range(1, self.num_features+1)]
            feature_vector = list(map(lambda x: x.mean(),feature_vector))
            feature_vector = softmax(feature_vector)

            label = 0
            y = -1
            if org_context in contexts[doc_index]:
                y = list(contexts[doc_index]).index(org_context)
            if y!=-1 and y < self.kbound:
                label = 1

            train_x.append(feature_vector)
            train_y.append(label)

        logistic = LogisticRegression()
        logistic.fit(train_x, train_y)

        return logistic

    def _get_logistic_regression(self):
        save_dir = p.join(self.args.path.embed, self.args.model.retriever_name)
        logistic_path = p.join(save_dir, "classifier.bin")
        if not p.exists(save_dir):
            os.mkdir(save_dir)
        if p.isfile(logistic_path):
            with open(logistic_path, "rb") as f:
                self.logistic = pickle.load(f)
        else:
            self.logistic = self._exec_logistic_regression()
            with open(logistic_path, "wb") as f:
                pickle.dump(self.logistic, f)

    def get_relevant_doc_bulk(self, queries, topk):
        min_topk = pow(2,self.num_features)
        
        dense_scores, dense_indices = self.dense_retriever.get_relevant_doc_bulk(queries, topk)
        sparse_scores, sparse_indices = self.sparse_retriever.get_relevant_doc_bulk(queries, max(min_topk, topk))
        sparse_scores = np.array(sparse_scores)

        feature_vectors=[]
        for sparse_score in sparse_scores:
            feature_vector = [sparse_score[:pow(2, i)] for i in range(1, self.num_features+1)]
            feature_vector = list(map(lambda x: x.mean(),feature_vector))
            feature_vector = softmax(feature_vector)
            feature_vectors.append(feature_vector)

        labels = self.logistic.predict(feature_vectors)

        doc_scores, doc_indices = [], []
        for k in range(topk):
            if labels[k]==1:
                doc_scores.append(sparse_scores[k])
                doc_indices.append(sparse_indices[k])
            else:
                doc_scores.append(dense_scores[k])
                doc_indices.append(dense_indices[k])

        return doc_scores, doc_indices

