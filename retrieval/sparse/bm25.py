import numpy as np
from retrieval.sparse import SparseRetrieval
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

class BM25(SparseRetrieval):
    def __init__(self, args, tokenize_fn, context_path="wikipedia_documents.json", b=0.75, k1=1.2):
        super().__init__(args, tokenize_fn=tokenize_fn, context_path=context_path)
        self.name = "BM25"
        self.b = b
        self.k1 = k1
        self.tfidfv = TfidfVectorizer(tokenizer=tokenize_fn, ngram_range=(1, 2), max_features=50000)
        self.avdl = None

    def get_sparse_embedding(self):
        super().get_sparse_embedding()
        self.tfidfv.fit(self.contexts)
        self.p_embedding = self.tfidfv.transform(self.contexts)
        self.avdl = self.p_embedding.sum(1).mean()

    def get_relevant_doc(self, query, k=1):
        query_vec = self.tfidfv.transform([query])
        assert np.sum(query_vec) != 0, "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        b, k1, avdl = self.b, self.k1, self.avdl

        len_p = self.p_embedding.sum(1).A1

        p_emb_for_q = self.p_embedding.tocsc()[:, query_vec.indices]
        denom = p_emb_for_q + (k1 * (1 - b + b * len_p / avdl))[:, None]

        # idf(t) = log [ n / df(t) ] + 1 in sklearn, so it need to be converted
        # to idf(t) = log [ n / df(t) ] with minus 1
        idf = self.tfidfv._tfidf.idf_[None, query_vec.indices] - 1.
        numer = p_emb_for_q.multiply(np.broadcast_to(idf, p_emb_for_q.shape)) * (k1 + 1)

        result = (numer / denom).sum(1).A1

        if not isinstance(result, np.ndarray):
            result = result.toarray()

        sorted_result_idx = np.argsort(result)[::-1]
        return result[sorted_result_idx].tolist()[:k], sorted_result_idx.tolist()[:k]

    def get_relevant_doc_bulk(self, queries, k=1):
        q = super(TfidfVectorizer, self.tfidfv).transform(queries)
        assert np.sum(q) != 0, "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."
        doc_scores = []
        doc_indices = []

        for query in queries:
            doc_score, doc_indice = self.get_relevant_doc(query, k)
            doc_scores.append(doc_score)
            doc_indices.append(doc_indice)

        return doc_scores, doc_indices