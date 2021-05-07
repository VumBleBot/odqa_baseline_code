import numpy as np
from collections import defaultdict
from retrieval.dense import DprRetrieval,DprKobertRetrieval,DprKorquadBertRetrieval
from retrieval.sparse import BM25Retrieval,TfidfRetrieval
import pandas as pd
from tqdm.auto import tqdm
from datasets import Dataset
from datasets import Sequence, Value, Features, DatasetDict

class HybridRetrieval(DprRetrieval):
    def __init__(self,args):
        super().__init__(args)
        self.args=args
        self.dense_retrieval=None
        self.sparse_retrieval=None

    def _get_retriever(self,args):
        RETRIEVER = {
        "DPR": DprRetrieval,
        "BM25": BM25Retrieval,
        "TFIDF": TfidfRetrieval,
        "DPRKOBERT": DprKobertRetrieval,
        "DPRKORQUAD": DprKorquadBertRetrieval,
        }
        retriever = RETRIEVER[args.model.retriever_name](args)
        retriever.get_embedding()
        return retriever

    def _prepare(self):
        # Get sparse/dense retriever at same condition
        args=self.args
        args.model.retriever_name = args.retriever.sparse_retriever_name
        self.sparse_retrieval = self._get_retriever(args)
        args.model.retriever_name = args.retriever.dense_retriever_name
        self.dense_retrieval = self._get_retriever(args)
        args.model.retriever_name = 'HYBRID'

    def _hybrid_results(self,dense_hits, sparse_hits, alpha, topk):
        hybrid_result=[]
        min_dense_score=min(dense_hits.values())
        min_sparse_score=min(sparse_hits.values())
        for doc in set(dense_hits.keys()) | set(sparse_hits.keys()):
            if doc not in dense_hits:
                score = alpha * sparse_hits[doc] + min_dense_score
            elif doc not in sparse_hits:
                score = alpha * min_sparse_score + dense_hits[doc]
            else:
                score = alpha * sparse_hits[doc] + dense_hits[doc]
            hybrid_result.append((doc,score))
        return sorted(hybrid_result, key=lambda x: x[1], reverse=True)[:topk]

    # Baseretireval의 method를 오버라이딩 해야함
    # query_or_dataset : datasets["validation"]
    def retrieve(self, query_or_dataset, topk=1): 
        assert self.p_embedding is not None, "get_embedding()을 먼저 수행한 후에 retrieve()를 작동시켜 주세요. "
        self._prepare()
        topk = self.args.retriever.topk
        alpha = self.args.retriever.alpha

        dense_doc_scores, dense_docs_indices = self.dense_retrieval.get_relevant_doc_bulk(query_or_dataset["question"], topk) 
        sparse_doc_scores, sparse_docs_indices = self.sparse_retrieval.get_relevant_doc_bulk(query_or_dataset["question"], topk)
        
        dense_hit, sparse_hit = {}, {}
        doc_scores, doc_indices = [], []
        for idx,query_id in enumerate(query_or_dataset['id']):
            score_vec = dense_doc_scores[idx]
            indices_vec = dense_docs_indices[idx]
            dense_hits={}
            for i in range(topk):
                dense_hits[indices_vec[i]]=score_vec[i]

            score_vec = sparse_doc_scores[idx]
            indices_vec = sparse_docs_indices[idx]
            sparse_hits={}
            for i in range(topk):
                sparse_hits[indices_vec[i]]=score_vec[i]
            
            hybrid_results=self._hybrid_results(dense_hits, sparse_hits, alpha, topk)

            doc_score,doc_indice=[],[]
            for _id, score in hybrid_results:
                doc_score+=[score]
                doc_indice+=[_id]
            doc_scores.append(doc_score)
            doc_indices.append(doc_indice)
        total = []
        for idx, example in enumerate(tqdm(query_or_dataset, desc="Retrieval: ")):
            for doc_id in range(topk):
                tmp = {
                    "question": example["question"],
                    "id": example["id"],
                    "context_id": doc_indices[idx][doc_id],  # retrieved id
                    "context": self.contexts[doc_indices[idx][doc_id]],  # retrieved doument
                }
                if "context" in example.keys() and "answers" in example.keys():
                    tmp["original_context"] = example["context"]  # original document
                    tmp["answers"] = example["answers"]  # original answer
                total.append(tmp)

        df = pd.DataFrame(total)

        if self.args.train.do_predict is True:
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
                    "original_context": Value(dtype="string", id=None),
                }
            )

        datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
        return datasets

