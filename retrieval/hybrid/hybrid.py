from retrieval.dense import DprBert
from retrieval.hybrid import HybridRetrieval, HybridLogisticRetrieval
from retrieval.sparse import TfidfRetrieval, BM25Retrieval, ATIREBM25Retrieval


class Bm25DprBert(HybridRetrieval):
    def __init__(self, args):
        super().__init__(args)
        temp = args.model.retriever_name

        args.model.retriever_name = "BM25"
        self.sparse_retriever = BM25Retrieval(args)
        args.model.retriever_name = "DPRBERT"
        self.dense_retriever = DprBert(args)

        args.model.retriever_name = temp


class TfidfDprBert(HybridRetrieval):
    def __init__(self, args):
        super().__init__(args)
        temp = args.model.retriever_name

        args.model.retriever_name = "TFIDF"
        self.sparse_retriever = TfidfRetrieval(args)
        args.model.retriever_name = "DPRBERT"
        self.dense_retriever = DprBert(args)

        args.model.retriever_name = temp


class AtireBm25DprBert(HybridRetrieval):
    def __init__(self, args):
        super().__init__(args)
        temp = args.model.retriever_name

        args.model.retriever_name = "ATIREBM25"
        self.sparse_retriever = ATIREBM25Retrieval(args)
        args.model.retriever_name = "DPRBERT"
        self.dense_retriever = DprBert(args)

        args.model.retriever_name = temp


class LogisticBm25DprBert(HybridLogisticRetrieval):
    def __init__(self, args):
        super().__init__(args)
        temp = args.model.retriever_name

        args.model.retriever_name = "BM25"
        self.sparse_retriever = BM25Retrieval(args)
        args.model.retriever_name = "DPRBERT"
        self.dense_retriever = DprBert(args)

        args.model.retriever_name = temp


class LogisticAtireBm25DprBert(HybridLogisticRetrieval):
    def __init__(self, args):
        super().__init__(args)
        temp = args.model.retriever_name

        args.model.retriever_name = "ATIREBM25"
        self.sparse_retriever = ATIREBM25Retrieval(args)
        args.model.retriever_name = "DPRBERT"
        self.dense_retriever = DprBert(args)

        args.model.retriever_name = temp
