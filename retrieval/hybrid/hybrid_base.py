from retrieval.base_retrieval import Retrieval


class HybridRetrieval(Retrieval):
    """ 이미 학습된 Sparse, Dense Retriever를 사용한다."""

    def __init__(self, args):
        """super().__init__(args) 생략, wiki를 가지고 있을 필요가 없음"""

        self.sparse_retriever = None
        self.dense_retriever = None

    def get_embedding(self):
        self.sparse_retriever.get_embedding()
        self.dense_retriever.get_embedding()

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

    def get_relevant_doc_bulk(self, query_or_dataset, topk):
        assert self.p_embedding is not None, "get_embedding()을 먼저 수행한 후에 retrieve()를 작동시켜 주세요. "

        dense_scores, dense_indices = self.dense_retriever.get_relevant_doc_bulk(query_or_dataset["question"], topk)
        sparse_scores, sparse_indices = self.sparse_retriever.get_relevant_doc_bulk(query_or_dataset["question"], topk)

        doc_scores, doc_indices = [], []

        for idx, query_id in enumerate(query_or_dataset["id"]):
            scores_topk, indexs_topk = dense_scores[idx], dense_indices[idx]
            dense_hits = {indexs_topk[k]: scores_topk[k] for k in range(topk)}

            scores_topk, indexs_topk = sparse_scores[idx], sparse_indices[idx]
            sparse_hits = {indexs_topk[k]: scores_topk[k] for k in range(topk)}

            doc_score, doc_index = self._rank_fusion_by_hybrid(dense_hits, sparse_hits)

            doc_scores.append(doc_score)
            doc_indices.append(doc_index)

        return doc_scores, doc_indices
