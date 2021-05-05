import os
import pickle
import os.path as p

import torch
import numpy as np

from retrieval.base_retrieval import Retrieval


class DenseRetrieval(Retrieval):
    def __init__(self, args):
        super().__init__(args)  # wiki context load
        self.name = args.model.retriever_name

        save_dir = p.join(args.path.embed, self.name)

        if not p.exists(save_dir):
            os.mkdir(save_dir)

        self.embed_path = p.join(save_dir, "embedding.bin")
        self.encoder_path = p.join(save_dir, f"{self.name}.pth")

        self.encoder = None
        self.p_embedding = None

    def _get_encoder(self):
        """ 모델 구조 가져오기 """
        raise NotImplementedError

    def _exec_embedding(self):
        raise NotImplementedError

    def get_embedding(self):
        if p.isfile(self.embed_path) and p.isfile(self.encoder_path) and not self.args.retriever.retrain:
            with open(self.embed_path, "rb") as f:
                self.p_embedding = pickle.load(f)

            self.encoder = self._get_encoder()
            self.encoder.load_state_dict(torch.load(self.encoder_path))
        else:
            self.p_embedding, self.encoder = self._exec_embedding()
            self.p_embedding.squeeze_()  # in-place
            self.p_embedding = self.p_embedding.detach().cpu().numpy()

            with open(self.embed_path, "wb") as f:
                pickle.dump(self.p_embedding, f)

            torch.save(self.encoder.state_dict())

    def get_relevant_doc_bulk(self, queries, k=1):
        self.encoder.eval()  # question encoder
        self.encoder.cuda()

        with torch.no_grad():
            q_seqs_val = self.tokenizer(queries, padding="max_length", truncation=True, return_tensors="pt").to("cuda")
            q_embedding = self.encoder(**q_seqs_val)
            q_embedding.squeeze_()  # in-place
            q_embedding = q_embedding.cpu().detach().numpy()

        # p_embedding: numpy, q_embedding: numpy
        doc_scores = np.matmul(q_embedding, self.p_embedding.T)
        doc_indices = np.argsort(doc_scores, axis=1)[:, -k:][:, ::-1]
        return doc_scores, doc_indices
