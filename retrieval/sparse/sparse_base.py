import os
import pickle
import os.path as p

from retrieval.base_retrieval import Retrieval


class SparseRetrieval(Retrieval):
    def __init__(self, args):
        super().__init__(args)

        self.name = args.model.retriever_name

        save_dir = p.join(args.path.embed, self.name)
        if not p.exists(save_dir):
            os.mkdir(save_dir)

        self.embed_path = p.join(save_dir, "embedding.bin")
        self.encoder_path = p.join(save_dir, f"{self.name}.bin")

        self.encoder = None
        self.p_embedding = None

    def _exec_embedding(self):
        raise NotImplementedError

    def get_embedding(self):
        if p.isfile(self.embed_path) and p.isfile(self.encoder_path) and not self.args.retriever.retrain:
            with open(self.embed_path, "rb") as f:
                self.p_embedding = pickle.load(f)

            with open(self.encoder_path, "rb") as f:
                self.encoder = pickle.load(f)
        else:
            self.p_embedding, self.encoder = self._exec_embedding()

            with open(self.embed_path, "wb") as f:
                pickle.dump(self.p_embedding, f)

            with open(self.encoder_path, "wb") as f:
                pickle.dump(self.encoder, f)

    def get_relevant_doc_bulk(self, queries, k=1):
        raise NotImplementedError
