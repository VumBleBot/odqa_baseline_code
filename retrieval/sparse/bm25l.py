import numpy as np
from retrieval.sparse import BM25Retrieval

class BM25LRetrieval(BM25Retrieval):
    def __init__(self, args):
        super().__init__(args)

        self.delta = 0.6

    def calculate_score(self, p_embedding, query_vec):
        b, k1, avdl, delta = self.b, self.k1, self.avdl, self.delta
        len_p = self.dls

        p_emb_for_q = p_embedding[:, query_vec.indices]
        ctd = p_emb_for_q / (1 - b + b * len_p / avdl)[:, None]
        denom = k1 + (ctd + delta)

        # idf(t) = log [ n / df(t) ] + 1 in sklearn, so it need to be converted
        # to idf(t) = log [ n / df(t) ] with minus 1
        idf = self.idf[None, query_vec.indices] - 1.0

        numer = np.multiply((ctd + delta), np.broadcast_to(idf, p_emb_for_q.shape)) * (k1 + 1)

        result = (numer / denom).sum(1).A1

        if not isinstance(result, np.ndarray):
            result = result.toarray()

        return result