# Retriever 구조

**현재 구현된 Retriever는 아래와 같이 네가지로 구분할 수 있습니다.**

- 단어 빈도수를 기반으로 문서를 검색하는 **SparseRetrieval**
- 잠재 벡터를 기반으로 문서를 검색하는 **DenseRetrieval**
- SparseRetrieval, DenseRetrieval를 같이 사용하는 **HybridRetrieval**
- LogisticRegression으로 SparseRetrieval와 DenseRetrieval중 하나를 선택하여 사용하는 **HybridLogisticRetrieval**

## Base Retriever

모든 Retriever가 참조하는 클래스입니다. Retriever의 `retrieve`기능이 구현되어 있습니다.

### Sparse Retriever

단어 빈도수를 기반으로 문서를 검색하는 SparseRetrieval입니다. `sklearn`에서 제공하는 `TfidfVectorizer`를 사용하여 `encoder`를 학습합니다.

document score를 어떻게 계산하냐에 따라서 여러가지 Retriever를 만들 수 있습니다.

- tfidf 
- bm25plus
- bm25l
- atirebm25

### Dense Retriever

잠재 벡터를 기반으로 문서를 검색하는 DenseRetrieval입니다.

`CLS Token`을 사용하여 Context Level 유사도 기반으로 문서를 검색하거나 `Word Token`을 사용하여 Token Level 유사도 기반으로 문서를 검색합니다.

- DPR ( Dense Passage Retriever )
- ColBERT

> 데이터셋에 따라서 학습하는 과정이 달라지므로 `train()` 메소드를 mixin으로 구현했습니다.

- Dataset
    - BaseTrainMixin : 기존 데이터 셋, in-batch 학습을 진행합니다.
    - Bm25TrainMixin : Bm25 Score를 기준으로 negative context가 추가된 데이터 셋

### HybridRetrieval

SparseRetrieval, DenseRetrieval를 같이 사용하는 HybridRetrieval입니다. sparse score와 dense score를 rank_fusion한 후 새로운 score를 기준으로 정렬한 문서를 반환합니다.

- TfidfDprBert
- Bm25DprBert
- AtireBm25DprBert

### HybridLogisticRetrieval

LogisticRegression으로 SparseRetrieval와 DenseRetrieval중 하나를 선택하여 사용하는 HybridLogisticRetrieval입니다. SparseRetrieval의 TOP-K Score에서 Feature Vector를 생성한 후 LogisticRegression에서 SparseRetrieval의 사용 유무를 결정합니다.

- LogisticBm25DprBert
- LogisticAtireBm25DprBert
