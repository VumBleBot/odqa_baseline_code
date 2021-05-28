import pickle
import numpy as np
import os.path as p
from tqdm.auto import tqdm

from konlpy.tag import Mecab
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

from retrieval.sparse import SparseRetrieval
from utils.tokenization_kobert import KoBertTokenizer


class BM25Retrieval(SparseRetrieval):
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
        self.encoder = TfidfVectorizer(tokenizer=self.tokenizer, ngram_range=(1, 2), use_idf=False, norm=None)
        self.idf_encoder = TfidfVectorizer(tokenizer=self.tokenizer, ngram_range=(1, 2), norm=None, smooth_idf=False)
        self.dls = np.zeros(len(self.contexts))

        for idx, context in enumerate(self.contexts):
            self.dls[idx] = len(context)

        self.avdl = np.mean(self.dls)
        self.p_embedding = None
        self.idf = None

    def get_embedding(self):
        if (
            p.isfile(self.embed_path)
            and p.isfile(self.encoder_path)
            and p.isfile(self.idf_encoder_path)
            and p.isfile(self.idf_path)
            and not self.args.retriever.retrain
        ):
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

    def calculate_idf(self):
        return self.idf_encoder.idf_

    def _exec_embedding(self):
        self.p_embedding = self.encoder.fit_transform(tqdm(self.contexts, desc="TF calculation: "))
        self.idf_encoder.fit(tqdm(self.contexts, desc="IDF calculation: "))
        self.idf = self.calculate_idf()

        return self.p_embedding, self.encoder, self.idf, self.idf_encoder

    def calculate_score(self, p_embedding, query_vec):
        raise NotImplementedError

    def get_relevant_doc_bulk(self, queries, topk):
        query_vecs = self.encoder.transform(queries)

        doc_scores = []
        doc_indices = []

        p_embedding = self.p_embedding.tocsc()

        self.results = []

        for query_vec in tqdm(query_vecs):

            result = self.calculate_score(p_embedding, query_vec)
            self.results.append(result)
            sorted_result_idx = np.argsort(result)[::-1]
            doc_score, doc_indice = result[sorted_result_idx].tolist()[:topk], sorted_result_idx.tolist()[:topk]
            doc_scores.append(doc_score)
            doc_indices.append(doc_indice)

        if not isinstance(self.results, np.ndarray):
            self.results = np.array(self.results)

        return doc_scores, doc_indices
