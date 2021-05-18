from retrieval.base_retrieval import Retrieval
from retrieval.sparse import ATIREBM25Retrieval, BM25LRetrieval, BM25PlusRetrieval
import numpy as np

class BM25EnsembleRetrieval(Retrieval):
    """ 이미 학습된 BM25 Retriever들을 사용한다."""

    def __init__(self, args):
        super().__init__(args)

        temp = args.model.retriever_name

        args.model.retriever_name = "ATIREBM25"
        args.retriever.b = 0.3
        args.retriever.k1 = 1.1
        self.atire_bm25 = ATIREBM25Retrieval(args)
        args.model.retriever_name = "BM25L"
        args.retriever.b = 0.3
        args.retriever.k1 = 1.8
        self.bm25l = BM25LRetrieval(args)
        args.model.retriever_name = "BM25Plus"
        args.retriever.b = 0.3
        args.retriever.k1 = 1.6
        self.bm25plus = BM25PlusRetrieval(args)

        args.model.retriever_name = temp

    def get_embedding(self):
        self.atire_bm25.get_embedding()
        self.bm25l.get_embedding()
        self.bm25plus.get_embedding()

        self.p_embedding = 1  # fake for super().retrieve's, assert line

    def get_relevant_doc_bulk(self, queries, topk):
        _, _ = self.atire_bm25.get_relevant_doc_bulk(queries, 1)
        _, _ = self.bm25l.get_relevant_doc_bulk(queries, 1)
        _, _ = self.bm25plus.get_relevant_doc_bulk(queries, 1)

        ensemble_results = (self.atire_bm25.results + self.bm25l.results + self.bm25plus.results)/3

        doc_scores, doc_indices = [], []

        if not isinstance(ensemble_results, np.ndarray):
            ensemble_results = ensemble_results.toarray()

        for i in range(ensemble_results.shape[0]):
            sorted_result = np.argsort(ensemble_results[i, :])[::-1]
            doc_scores.append(ensemble_results[i, :][sorted_result].tolist()[:topk])
            doc_indices.append(sorted_result.tolist()[:topk])

        return doc_scores, doc_indices
