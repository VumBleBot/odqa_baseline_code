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
        self.tokenizer = None
        self.p_embedding = None

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

    def _load_dataset(self):
        """학습에 필요한 데이터를 불러옵니다. 데이터셋에 따라서 전처리가 달라집니다.
        Returns:
            train_dataset: TensorDataset of ['train_dataset', 'bm25_document_questions', 'bm25_question_documents']
        """
        raise NotImplementedError

    def _exec_embedding(self):
        """Training Argument를 지정한 후 학습을 진행합니다.
        Returns:
            p_encoder: trained passage encoder
            q_encoder: trained question encoder
        """
        raise NotImplementedError

    def _train(self):
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

    def get_relevant_doc_bulk(self, queries, topk=1):
        self.encoder.eval()  # question encoder
        self.encoder.cuda()

        with torch.no_grad():
            q_seqs_val = self.tokenizer(
                queries, padding="longest", truncation=True, max_length=512, return_tensors="pt"
            ).to("cuda")
            q_embedding = self.encoder(**q_seqs_val)
            q_embedding.squeeze_()  # in-place
            q_embedding = q_embedding.cpu().detach().numpy()

        # p_embedding: numpy, q_embedding: numpy
        result = np.matmul(q_embedding, self.p_embedding.T)
        doc_indices = np.argsort(result, axis=1)[:, -topk:][:, ::-1]
        doc_scores = []

        for i in range(len(doc_indices)):
            doc_scores.append(result[i][[doc_indices[i].tolist()]])

        return doc_scores, doc_indices
