from prepare import get_dataset, get_retriever
import numpy as np
from collections import defaultdict
from retrieval.dense import DprRetrieval,DenseRetrieval
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

    def _prepare(self):
        # In same condition, except retriever name
        args.model.retriever_name="DPR"
        self.dense_retrieval = get_retriever(args)
        args.model.retriever_name="BM25"
        self.sparse_retrieval = get_retriever(args)

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

    # Baseretireval의 method를 오버라이딩 하는게 관건임 
    def retrieve(self, alpha: float=0.1): # alpha : hyper-param
        assert self.p_embedding is not None, "get_embedding()을 먼저 수행한 후에 retrieve()를 작동시켜 주세요. "
        self._prepare()
        datasets = get_dataset(args, is_train=True)

        topk = args.retriever.topk

        # valid_datasets = self.dense_retrieval.retrieve(d["validation"], topk=topk)
        dense_doc_scores, dense_docs_indices = self.dense_retrieval.get_relevant_doc_bulk(datasets["validation"]["question"], topk) 
        # dense_hits = { docs_indices[i]:doc_scores[i] for i in range(len(doc_scores))}

        # valid_datasets = self.sparse_retrieval.retrieve(datasets["validation"], topk=args.retriever.topk)
        sparse_doc_scores, sparse_docs_indices = self.sparse_retrieval.get_relevant_doc_bulk(datasets["validation"]["question"], topk)
        # sparse_hits = { docs_indices[i]:doc_scores[i] for i in range(len(doc_scores))}

        # row - 쿼리는 변동 없음. 
        hybrid_doc_scores = np.hstack([dense_doc_scores,sparse_doc_scores])
        hybrid_doc_indices = np.hstack([dense_docs_indices,sparse_docs_indices])


        dense_hit={}
        sparse_hit={}
        doc_scores=[]
        doc_indices=[]
        for idx,query_id in enumerate(datasets['validation']['id']):
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
            print(hybrid_results)

            doc_score,doc_indice=[],[]
            for _id, score in hybrid_results:
                doc_score+=[score]
                doc_indice+=[_id]
            doc_scores.append(doc_score)
            doc_indices.append(doc_indice)

        total = []
        for idx, example in enumerate(tqdm(datasets["validation"], desc="Retrieval: ")):
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

if __name__=="__main__":
    from tools import get_args
    args=get_args()
    clss=HybridRetrieval(args)
    clss.get_embedding()
    clss.retrieve()