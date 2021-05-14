# Retriver 구조

- retrieval
    - dense
    - hybrid
    - sparse


**Depth는 상속 Depth를 의미합니다.** 표현이 좀 이상하긴 한데 찰떡같이 이해하실거라고 믿습니다 ㅎㅎ

## [Depth 1] Base Retriver 설명

- retrieve(self, query_or_dataset, topk=1)
    - 검색된 topk개의 문서를 DatasetDict 형식으로 반환합니다.
    - 240개의 문서를 검색하고 topk가 3이라면, 총 720개의 데이터가 반환이 됩니다.
    - `question`, `context`, `기타`... 형식

### [Depth 2] Sparse Retriver 설명

- get\_embedding: 파일이 경로에 존재할 경우 사용, 없을 경우 학습을 진행합니다.
    - 파일: 'embed.bin', 'encoder.bin'

#### [Depth 3] Tfidf Retriver 설명

- \_exec_embedding: TFIDF 알고리즘으로 wiki 데이터 셋을 학습합니다.
- - get_relevant_doc_bulk: TFIDF 점수를 기준으로 question에 맞는 document들을 top-k개 뽑아옵니다.

#### [Depth 3] Bm25 Retriver 설명

- \_exec_embedding: TFIDF 알고리즘으로 wiki 데이터 셋을 학습합니다.
    - BM25는 TFIDF 점수는 위와 동일하나 Score 계산을 다르게 합니다.
- \_get_relevant_doc_bulk: TFIDF 점수와 BM25 알고리즘을 기준으로 question에 맞는 document들을 top-k개 뽑아옵니다.

### [Depth 2] Dense Retriver 설명

- \_get_embedding: 파일이 경로에 존재할 경우 사용, 없을 경우 학습을 진행합니다.
    - 파일: 'embed.bin', 'encoder.pth'
- \_get_relevant_doc_bulk question_vector와 context_vector의 유사도를 계산해 반환합니다.

#### [Depth 3] dpr base 설명

> DPR을 학습하는 모듈이라고 보시며 될 것 같습니다.

- get_retriever_dataset: retriever 데이터 셋을 가져옵니다. prepare.py의 get_dataset과 비슷하지만 따로 구현을 했습니다. > 리팩토링 가능성 굉장히 높습니다.
- epoch_time: epoch 시간을 계산해주는 함수입니다.
- DprRetrieval: 
    - 모델을 불러오고
    - 데이터 셋을 준비합니다.
    - 그리고 학습을 진행합니다.
    - 학습이 완료된 후 context 임베딩 벡터를 생성합니다.

- BaseTrainMixin
    - \_load_dataset: `train_dataset` 데이터 셋을 준비합니다.
    - \_train: `train_dataset` 데이터 셋으로 학습합니다.

- Bm25TrainMixin
    - \_load_dataset: `bm25로 구축된` 데이터 셋을 준비합니다.
    - \_train: `bm25로 구축된` 데이터 셋으로 학습합니다.

> mixin은 `get_retriever`에서 Rretrieval을 호출할 때 `dataset`을 기준으로 적용됩니다.

#### [Depth 4] dpr 설명

- BertEncoder: `다국어 Bert` 인코더입니다.
- DprBert: `다국어 Bert`를 기반으로 DPR 모델입니다.

### [Depth 2] hybrid_base.py 설명

- \_rank_fusion_by_hybrid: sparse와 dense의 score를 사용하여 새로운 score를 계산합니다. 그리고 정렬해서 반환합니다.
- \_get_relevant_doc_bulk: sparse와 dense의 결과를 조합하여 반환합니다.

#### [Depth 3] hybrid.py 설명

- Bm25DprBert: Bm25와 DprBert를 사용합니다.
- TfidfDprBert:  Tfidf와 DprBert를 사용합니다.
