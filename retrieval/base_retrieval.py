from datasets import Sequence, Value, Features, DatasetDict

from tqdm.auto import tqdm
import pandas as pd
import pickle
import json
import os
import numpy as np
import torch

from datasets import Dataset


class Retrieval:
    def __init__(self, args, tokenize_fn, context_path="wikipedia_documents.json"):
        self.args=args
        self.name = None # eg. tfidf, bm25, dpr
        self.embedding_arc = None # TfidfVectoizer or BertEncoder etc 
        self.p_embedding = None

        with open(os.path.join(self.args.data_path, "data", context_path), "r") as f:
            wiki = json.load(f)
            
        self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))  # 데이터셋에는 56737개의 unique contexts 존재
        self.ids = list(range(len(self.contexts)))

    def exec_embedding(self): # tf-idf의 경우는 vecotrizer, 다른 애는 
        return 

    def get_embedding(self):
        """
            bin으로 저장해두었다면 불러오거나, 새로 embedding을 하는 애.
        """
        pickle_name = "embedding.bin"
        embedding_arc_name = f"{self.name}.bin"

        embed_dir = os.path.join(self.args.path.embed, self.name.upper())
        if not os.path.exists(embed_dir):
            os.mkdir(embed_dir)
        
        emd_path = os.path.join(embed_dir, pickle_name)
        embedding_arc_path = os.path.join(embed_dir, embedding_arc_name) # tfidfv/dpr 경로 ... 

        # sparse에서는 pickle로 넘겨도 되는데, dense에서는 pth를 넣어야 한다. 피클로 넣을 이유가 없음
        # 여기서 꼬임. => argc를 어떻게 넣어? 
        if self.name in ('tfidf','bm25'):
            if os.path.isfile(emd_path) and os.path.isfile(embedding_arc_path):
                with open(emd_path, "rb") as file:
                    self.p_embedding = pickle.load(file)
                with open(embedding_arc_path, "rb") as file:
                    self.embedding_arc = pickle.load(file)
            else: 
                self.p_embedding, self.embedding_arc = self.exec_embedding()
                with open(emd_path, "wb") as file:
                    pickle.dump(self.p_embedding, file)
                with open(embedding_arc_path, "wb") as file:
                    pickle.dump(self.embedding_arc, file)
        else: # dense인 애들의 경우는 BertEncoder가 embedding_arc이라 저장할 수 없고, get_embedding 부분에서 
            if os.path.isfile(emb_path):
                with open(emd_path, "rb") as file:
                    self.p_embedding = pickle.load(file) 
            else:
                self.p_embedding, _ = self.exec_embedding()  # p_encoder은 굳이 필요없음  
                with open(emd_path, "wb") as file:
                    pickle.dump(self.p_embedding, file)
            self.p_embedding=torch.Tensor(self.p_embedding).squeeze() # (num_passage, emb_dim) 
    
    def get_relevant_doc_bulk(self, queries, k=1):
        print('*'*30)
        return 

    def retrieve_pipeline(self, args, query_or_dataset, topk=1):
        df = self.retrieve(query_or_dataset, topk=topk)

        if args.train.do_predict is True:
            f = Features(
                {
                    "context": Value(dtype="string", id=None),
                    "id": Value(dtype="string", id=None),
                    "question": Value(dtype="string", id=None),
                }
            )
        else:
            f = Features(
                {
                    "answers": Sequence(
                        feature={"text": Value(dtype="string", id=None), "answer_start": Value(dtype="int32", id=None)},
                        length=-1,
                        id=None,
                    ),
                    "context": Value(dtype="string", id=None),
                    "id": Value(dtype="string", id=None),
                    "question": Value(dtype="string", id=None),
                }
            )

        datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
        return datasets

    def retrieve(self, query_or_dataset, topk=1): # buld query만을 가정 (query_or_dataset, Dataset, topk)
        assert (
            self.p_embedding is not None
        ), "You must build faiss by self.get_sparse_embedding() before you run self.retrieve()." # (Question)
        
        total = []
        doc_scores, doc_indices = self.get_relevant_doc_bulk(query_or_dataset["question"], k=1)

        for idx, example in enumerate(tqdm(query_or_dataset, desc="Sparse retrieval: ")):
            tmp = {
                "question": example["question"],
                "id": example["id"],
                "context_id": doc_indices[idx][0],  # retrieved id
                "context": self.contexts[doc_indices[idx][0]],  # retrieved doument
            }
            if "context" in example.keys() and "answers" in example.keys():
                tmp["original_context"] = example["context"]  # original document
                tmp["answers"] = example["answers"]  # original answer
            total.append(tmp)

        cqas = pd.DataFrame(total)
        return cqas

