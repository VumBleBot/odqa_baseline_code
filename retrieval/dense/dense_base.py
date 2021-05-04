import pickle
import os.path as p

import torch
import numpy as np

from retrieval.base_retrieval import Retrieval


class DenseRetrieval(Retrieval):
    def __init__(self, args):
        super().__init__(args)  # wiki context load
        self.name = args.model.retriever_name

        self.embed_path = p.join(args.path.embed, "embedding.bin")
        self.encoder_path = p.join(args.path.embed, f"{self.name}.pth")

        self.encoder = None
        self.p_embedding = None

    def _exec_embedding(self):
        raise NotImplementedError

    def get_embedding(self):
        if p.isfile(self.embed_path) and p.isfile(self.encoder_path):
            with open(self.embed_path, "rb") as f:
                self.p_embedding = pickle.load(f)

            with open(self.encoder_path, "rb") as f:
                self.encoder = pickle.load(f)
        else:
            self.p_embedding, self.encoder = self._exec_embedding()
            self.p_embedding.squeeze_()  # in-place
            self.p_embedding = self.p_embedding.detach().cpu().numpy()

            with open(self.embed_path, "wb") as f:
                pickle.dump(self.p_embedding, f)

            with open(self.encoder_path, "wb") as f:
                torch.save(self.encoder.state_dict())

    def get_relevant_doc_bulk(self, queries, k=1):
        self.encoder.eval()  # question encoder
        self.encoder.cuda()

        with torch.no_grad():
            q_seqs_val = self.tokenizer(queries, padding="max_length", truncation=True, return_tensors="pt").cuda()
            q_embedding = self.encoder(**q_seqs_val)
            q_embedding.squeeze_()  # in-place
            q_embedding = q_embedding.detach().cpu().numpy()

        # p_embedding: numpy, q_embedding: numpy
        doc_scores = np.matmul(q_embedding, self.p_embedding.T)  # TODO: 수정 필요
        doc_indices = np.argsort(doc_scores, axis=1)[:, -k:]
        return doc_scores, doc_indices