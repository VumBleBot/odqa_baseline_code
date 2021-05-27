import numpy as np

from konlpy.tag import Mecab
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

from retrieval.sparse import SparseRetrieval
from utils.tokenization_kobert import KoBertTokenizer


class TfidfRetrieval(SparseRetrieval):
    def __init__(self, args):
        super().__init__(args)

        if self.args.model.tokenizer_name == "":
            print("Using Mecab tokenizer")
            mecab = Mecab()
            self.tokenizer = mecab.morphs
        elif self.args.model.tokenizer_name in ["monologg/kobert", "monologg/distilkobert"]:
            print("Using KoBert tokenizer")
            self.tokenizer = KoBertTokenizer.from_pretrained(args.model.tokenizer_name).tokenize
        else:
            print("Using AutoTokenizer: ", args.model.tokenizer_name)
            self.tokenizer = AutoTokenizer.from_pretrained(args.model.tokenizer_name, use_fast=True).tokenize

        mecab = Mecab()
        self.encoder = TfidfVectorizer(tokenizer=self.tokenizer, ngram_range=(1, 2))
        self.p_embedding = None

    def _exec_embedding(self):
        p_embedding = self.encoder.fit_transform(self.contexts)
        return p_embedding, self.encoder

    def get_relevant_doc_bulk(self, queries, topk):
        query_vec = self.encoder.transform(queries)
        assert np.sum(query_vec) != 0, "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        result = query_vec * self.p_embedding.T

        if not isinstance(result, np.ndarray):
            result = result.toarray()

        doc_scores, doc_indices = [], []

        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:topk])
            doc_indices.append(sorted_result.tolist()[:topk])

        return doc_scores, doc_indices
