import tqdm 
import numpy as np
from retrieval.base_retrieval import Retrieval
from sklearn.linear_model import LogisticRegression
from scipy.special import softmax
from datasets import load_from_disk

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
            scores_topk, indexs_topk = dense_scores[idx], dense_indices[idx]
            dense_hits = {indexs_topk[k]: scores_topk[k] for k in range(topk)}

            scores_topk, indexs_topk = sparse_scores[idx], sparse_indices[idx]
            sparse_hits = {indexs_topk[k]: scores_topk[k] for k in range(topk)}

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

    def get_embedding(self):
        self.sparse_retriever.get_embedding()
        self.dense_retriever.get_embedding()
        self.logistic = LogisticRegression()
        self.p_embedding = 1  # fake for super().retrieve's, assert line

    def _exec_logistic_regression(self, queries, topk, sparse_scores, sparse_indices):
        train_dataset = load_from_disk(p.join(args.path.train_data_dir, 'train_dataset'))
        train_x, train_y = [], [] 
        num_features = args.retriever.num_features

        sparse_scores = np.array(sparse_scores)
        sparse_indices = np.array(sparse_indices)
        corpus = np.array(sparse_retriever.contexts)

        min_index = pow(2, num_features)

        for idx in tqdm.tqdm(range(len(sparse_scores))):
            doc_index = sparse_indices[idx]
            org_context = train_dataset['context'][idx]
            
            feature_vector = [sparse_scores [idx][:pow(2, i)] for i in range(1, num_features+1)]
            feature_vector = list(map(lambda x: x.mean(),feature_vector))
            feature_vector = softmax(feature_vector)
                    
            y = list(corpus[doc_index]).find(org_context)

            class_ = 0
            if y!=-1 and y < topk:
                class_=1

            train_x.append(feature_vector)
            train_y.append(class_)

        self.logistic.fit(train_x, train_y)

        return train_x, train_y

    def _get_logistic_regression(self, train_x, threshold):
        return self.logistic.predict(train_x)
        # return self.logistic.predict_proba(train_x[:,1] >= threshold).astype(bool)

    def get_relevant_doc_bulk(self, queries, topk):
        threshold = args.retriever.threshold
        dense_scores, dense_indices = self.dense_retriever.get_relevant_doc_bulk(queries, topk)
        sparse_scores, sparse_indices = self.sparse_retriever.get_relevant_doc_bulk(queries, topk)

        train_x, train_y = self._exec_logistic_regression(queries, topk, sparse_scores, sparse_indices)
        labels = self._get_logistic_regression(train_x, threshold)

        doc_scores, doc_indices = [], []
        for i in range(len(sparse_scores)):
            if labels[i]==1:
                doc_scores[i],doc_indices[i] = sparse_scores[i], sparse_indices[i]
            else:
                doc_scores[i],doc_indices[i] = dense_scores[i], dense_indices[i]

        return doc_scores, doc_indices







# get imbedding에서 가져오기 
# logistic 모델 가져오기
# threshold 가져오기 
# 일단 sparse랑 dense 모두 가져오고, get_relevant_doc_bulk 수행한다음에 
# 피처수랑 topk는 하이퍼파라미터. threshold까지 
# retrieve하면 embedding까지 됨. 

# # 학습
# log_reg.fit(train_x, train_y)
# # 예측 
# pred_y = (self.log_reg.predict_proba(train_x)[:,1] >= threshold).astype(bool)