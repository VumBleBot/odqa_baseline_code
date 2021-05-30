import numpy as np
from retrieval.sparse import BM25Retrieval

class BM25PlusRetrieval(BM25Retrieval):
    def __init__(self, args):
        super().__init__(args)
        self.delta = 0.7

    def calculate_idf(self):
        idf = self.idf_encoder.idf_
        idf = idf - np.log(len(self.contexts)) + np.log(len(self.contexts) + 1.0)
        return idf

    def calculate_score(self, p_embedding, query_vec):
        b, k1, avdl, delta = self.b, self.k1, self.avdl, self.delta
        len_p = self.dls

        p_emb_for_q = p_embedding[:, query_vec.indices]
        denom = p_emb_for_q + (k1 * (1 - b + b * len_p / avdl))[:, None]

        # idf(t) = log [ n / df(t) ] + 1 in sklearn, so it need to be converted
        # to idf(t) = log [ n / df(t) ] with minus 1
        idf = self.idf[None, query_vec.indices] - 1.0
        idf_broadcasted = np.broadcast_to(idf, p_emb_for_q.shape)

        numer = p_emb_for_q * (k1 + 1)

        result = (np.multiply((numer / denom) + delta, idf_broadcasted)).sum(1).A1

        if not isinstance(result, np.ndarray):
            result = result.toarray()

        return result