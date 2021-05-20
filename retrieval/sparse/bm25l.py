from tqdm.auto import tqdm
import pickle
import numpy as np
import os.path as p
from konlpy.tag import Mecab
from sklearn.feature_extraction.text import TfidfVectorizer
from tokenization_kobert import KoBertTokenizer
from transformers import AutoTokenizer

from retrieval.sparse import SparseRetrieval


class BM25LRetrieval(SparseRetrieval):
    def __init__(self, args):
        super().__init__(args)

        save_dir = p.join(args.path.embed, self.name)
        self.encoder_path = p.join(save_dir, f"{self.name}.bin")
        self.idf_encoder_path = p.join(save_dir, f"{self.name}_idf.bin")
        self.idf_path = p.join(save_dir, "idf.bin")

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

        self.b = self.args.retriever.b
        self.k1 = self.args.retriever.k1
        self.delta = 0.6
        self.encoder = TfidfVectorizer(tokenizer=self.tokenizer, ngram_range=(1, 2), use_idf=False, norm=None)
        self.idf_encoder = TfidfVectorizer(tokenizer=self.tokenizer, ngram_range=(1, 2), norm=None)
        self.dls = np.zeros(len(self.contexts))

        for idx, context in enumerate(self.contexts):
            self.dls[idx] = len(context)

        self.avdl = np.mean(self.dls)
        self.p_embedding = None
        self.idf = None

    def get_embedding(self):
        if p.isfile(self.embed_path) and p.isfile(self.encoder_path) and p.isfile(self.idf_encoder_path) and p.isfile(self.idf_path) and not self.args.retriever.retrain:
            with open(self.embed_path, "rb") as f:
                self.p_embedding = pickle.load(f)

            with open(self.encoder_path, "rb") as f:
                self.encoder = pickle.load(f)

            with open(self.idf_encoder_path, "rb") as f:
                self.idf_encoder = pickle.load(f)

            with open(self.idf_path, "rb") as f:
                self.idf = pickle.load(f)
        else:
            self.p_embedding, self.encoder, self.idf, self.idf_encoder = self._exec_embedding()

            with open(self.embed_path, "wb") as f:
                pickle.dump(self.p_embedding, f)

            with open(self.encoder_path, "wb") as f:
                pickle.dump(self.encoder, f)

            with open(self.idf_path, "wb") as f:
                pickle.dump(self.idf, f)

            with open(self.idf_encoder_path, "wb") as f:
                pickle.dump(self.idf_encoder, f)

    def _exec_embedding(self):
        self.p_embedding = self.encoder.fit_transform(tqdm(self.contexts, desc="TF calculation: "))
        self.idf_encoder.fit(tqdm(self.contexts, desc="IDF calculation: "))
        self.idf = self.idf_encoder.idf_

        return self.p_embedding, self.encoder, self.idf, self.idf_encoder

    def get_relevant_doc_bulk(self, queries, topk):
        query_vecs = self.encoder.transform(queries)

        b, k1, avdl, delta = self.b, self.k1, self.avdl, self.delta
        len_p = self.dls

        doc_scores = []
        doc_indices = []

        p_embedding = self.p_embedding.tocsc()

        self.results = []

        for query_vec in tqdm(query_vecs):
            p_emb_for_q = p_embedding[:, query_vec.indices]
            ctd = p_emb_for_q / (1 - b + b * len_p / avdl)[:, None]
            denom = k1 + (ctd + delta)

            # idf(t) = log [ n / df(t) ] + 1 in sklearn, so it need to be converted
            # to idf(t) = log [ n / df(t) ] with minus 1
            idf = self.idf[None, query_vec.indices] - 1.0

            numer = np.multiply((ctd + delta),np.broadcast_to(idf, p_emb_for_q.shape)) * (k1 + 1)

            result = (numer / denom).sum(1).A1

            if not isinstance(result, np.ndarray):
                result = self.result.toarray()

            self.results.append(result)
            sorted_result_idx = np.argsort(result)[::-1]
            doc_score, doc_indice = result[sorted_result_idx].tolist()[:topk], sorted_result_idx.tolist()[:topk]
            doc_scores.append(doc_score)
            doc_indices.append(doc_indice)

        if not isinstance(self.results, np.ndarray):
            self.results = np.array(self.results)

        return doc_scores, doc_indices
