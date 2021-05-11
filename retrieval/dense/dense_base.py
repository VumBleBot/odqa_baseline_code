import os
import pickle
import os.path as p
from itertools import chain

import torch
import numpy as np
from torch.utils.data import TensorDataset

from prepare import get_retriever_dataset
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
        self.tokenizer = None
        self.p_embedding = None

    def _load_dataset(self):
        # dataset.features : ['query', 'negative_samples', 'label']
        dataset = get_retriever_dataset(self.args)

        corpus_size = len(dataset["negative_samples"][0])
        negative_samples = list(chain(*dataset["negative_samples"]))

        # query
        q_seqs = self.tokenizer(
            dataset["query"],
            padding="longest",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )

        # negative_samples
        p_seqs = self.tokenizer(
            negative_samples,
            padding="longest",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )

        embedding_size = p_seqs.shape[-1]

        for k in p_seqs.keys():
            p_seqs[k] = p_seqs[k].reshape(-1, corpus_size, embedding_size)

        train_dataset = TensorDataset(
            p_seqs["input_ids"],
            p_seqs["attention_mask"],
            p_seqs["token_type_ids"],
            q_seqs["input_ids"],
            q_seqs["attention_mask"],
            q_seqs["token_type_ids"],
            torch.tensor(dataset["label"]),
        )

        return train_dataset

    def _load_model(self):
        """학습 때 사용할 모델을 가져옵니다.
        Returns:
            p_encoder: passage encoder, passage to embedding vector
            q_encoder: question encoder, question to embedding vector
        """
        raise NotImplementedError

    def _get_encoder(self):
        """추론 때 사용할 q_encoder 모델의 구조를 가져옵니다.
        Returns:
            q_encoder: question encoder, question to embedding vector
        """
        raise NotImplementedError

    def _exec_embedding(self):
        """학습을 진행합니다.
        Returns:
            p_encoder: trained passage encoder
            q_encoder: trained question encoder
        """
        raise NotImplementedError

    def get_embedding(self):
        if p.isfile(self.embed_path) and p.isfile(self.encoder_path) and not self.args.retriever.retrain:
            with open(self.embed_path, "rb") as f:
                self.p_embedding = pickle.load(f)

            self.encoder = self._get_encoder()
            self.encoder.load_state_dict(torch.load(self.encoder_path))
        else:
            self.p_embedding, self.encoder = self._exec_embedding()

            with open(self.embed_path, "wb") as f:
                pickle.dump(self.p_embedding, f)

            torch.save(self.encoder.state_dict(), self.encoder_path)

    def get_relevant_doc_bulk(self, queries, k=1):
        self.encoder.eval()  # question encoder
        self.encoder.cuda()

        with torch.no_grad():
            q_seqs_val = self.tokenizer(
                queries, padding="max_length", truncation=True, max_length=512, return_tensors="pt"
            ).to("cuda")
            q_embedding = self.encoder(**q_seqs_val)
            q_embedding.squeeze_()  # in-place
            q_embedding = q_embedding.cpu().detach().numpy()

        # p_embedding: numpy, q_embedding: numpy
        result = np.matmul(q_embedding, self.p_embedding.T)
        doc_indices = np.argsort(result, axis=1)[:, -k:][:, ::-1]
        doc_scores = []
        for i in range(len(doc_indices)):
            doc_scores.append(result[i][[doc_indices[i].tolist()]])
        return doc_scores, doc_indices
