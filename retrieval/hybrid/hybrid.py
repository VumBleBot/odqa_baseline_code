from retrieval.hybrid import HybridRetrieval
from retrieval.dense import DprKobertRetrieval
from retrieval.sparse import TfidfRetrieval, BM25Retrieval


class Bm25DprKobert(HybridRetrieval):
    def __init__(self, args):
        temp = args.model.retriever_name

        args.model.retriever_name = "BM25"
        self.sparse_retriever = BM25Retrieval(args)
        args.model.retriever_name = "DPRKOBERT"
        self.dense_retriever = DprKobertRetrieval(args)

        args.model.retriever_name = temp


class TfidfDprKobert(HybridRetrieval):
    def __init__(self, args):
        temp = args.model.retriever_name

        args.model.retriever_name = "TFIDF"
        self.sparse_retriever = TfidfRetrieval(args)
        args.model.retriever_name = "DPRKOBERT"
        self.dense_retriever = DprKobertRetrieval(args)

        args.model.retriever_name = temp
