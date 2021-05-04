import pickle
import os.path as p

import torch
import numpy as np

from retrieval.base_retrieval import Retrieval


class DenseRetrieval(Retrieval):
    def __init__(self, args, name):
        self.name = name

        self.embed_path = p.join(args.path.embed, "embedding.bin")
        self.encoder_path = p.join(args.path.embed, f"{self.name}.pth")

        super().__init__(args)

        self.encoder = None
        self.p_embedding = None

    def exec_embedding(self):
        raise NotImplementedError

    def get_embedding(self):
        if p.isfile(self.embed_path) and p.isfile(self.encoder_path):
            with open(self.embed_path, "rb") as f:
                self.p_embedding = pickle.load(f)

            with open(self.encoder_path, "rb") as f:
                self.encoder = pickle.load(f)
        else:
            self.p_embedding, self.encoder = self.exec_embedding()

            with open(self.embed_path, "wb") as f:
                """ document embedding """
                pickle.dump(self.p_embedding.cpu().numpy(), f)

            with open(self.encoder_path, "wb") as f:
                torch.save(self.encoder.state_dict())

    def get_relevant_doc_bulk(self, queries, k=1):
        self.encoder.eval()

        with torch.no_grad():
            q_seqs_val = self.tokenizer(queries, padding="max_length", truncation=True, return_tensors="pt").to("cuda")
            q_embedding = self.encoder(**q_seqs_val).to("cpu").numpy()

        # 각각 임베딩 dot product해서 score구하기
        doc_scores = np.matmul(q_embedding, np.transpose(self.p_embedding, 0, 1))
        doc_indices = np.argsort(doc_scores, dim=1, descending=True).squeeze()

        return doc_scores, doc_indices
