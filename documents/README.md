# ODQA HELPER DOCUMENT

- [reader.md](reader.md): reader 구조에 대해 설명되어 있습니다.
- [retriever.md](retriever.md): retriever 구조에 대해 설명되어 있습니다.


## 사용 가능한 모델들

### READER

- DPR
- LSTM
- CNN
- CCNN
- CCNN_v2
- CNN_LSTM
- CCNN_EM
- CCNN_LSTM_EM

**EX) examples/reader.json**

-  **"reader_name": "DPR"**

### RETRIEVER

**SparseRetrieval** : 단어 빈도수로 문서를 검색하는 전통적인 방법의 Retriever 입니다.

**DenseRetrieval**  : 잠재 벡터로 문서를 검색하는 딥러닝 기법의 Retriever 입니다.

**HybridRetrieval** : SparseRetrieval와 DenseRetrieval의 문서 반환 결과를 Weighted Sum을 사용하여 재정렬한 후 반환하는 Retriever 입니다.

**HybridLogisticRetrieval** : SparseRetrieval와 DenseRetrieval중에 어떤 것을 사용하여 예측할 지 LogisticRegression으로 판단하는 Retriever 입니다.

- TFIDF ( SparseRetrieval )
- BM25L ( SparseRetrieval )
- BM25Plus ( SparseRetrieval )
- ATIREBM25 ( SparseRetrieval )
- BM25Ensemble ( SparseRetrieval )
- COLBERT ( DenseRetrieval )
- DPRBERT ( DenseRetrieval )
- DPRELECTRA ( DenseRetrieval )
- TFIDF_DPRBERT ( HybridRetrieval )
- BM25_DPRBERT ( HybridRetrieval )
- ATIREBM25_DPRBERT ( HybridRetrieval )
- LOG_BM25_DPRBERT ( HybridLogisticRetrieval )
- LOG_ATIREBM25_DPRBERT ( HybridLogisticRetrieval )

**EX) examples/retriever.json**

-  **"retriever_name": "TFIDF"**

## 전체적인 구조

모듈기준으로 프로젝트의 전체적인 구조를 설명하겠습니다.

### 실행 모듈

구현한 모듈들을 사용하는 실행 모듈들입니다.

간략하게 모듈 설명과 관련 모듈들을 적어봤습니다.

- `run.py` : RETRIVER와 READER모델을 결합해 ODQA성능을 평가하는 모듈입니다.
    - run\_mrc.py와 관련있는 모듈들
    - run\_retrieval.py와 관련있는 모듈들
- `run_mrc.py` : 문서 검색 없이 READER의 기계 독해 능력을 평가하는 모듈입니다.
    - reader/\*
    - prepare.py
        - get_dataset: 데이터 셋을 가져옵니다.
        - get_reader: READER 모델을 불러옵니다.
- `run_retrieval.py` : RETRIEVER의 검색 성능을 평가하는 모듈입니다.
    - retriever/\*
    - prepare.py
        - get_dataset: 데이터 셋을 가져옵니다.
        - get_retriever: RETRIEVER 모델을 불러옵니다.
- `predict.py` : test\_dataset에 대한 결과값을 예측하는 모듈입니다.
    - prepare.py
        - get_dataset: 
        - get_reader: READER 모델을 불러옵니다.
        - get_retriever: RETRIEVER 모델을 불러옵니다.


### run.py 전략 세팅

> `run.py`는 `run_mrc.py`와 `run_retrieval.py`에 사용하는 세팅을 같이 사용하시면 됩니다!

### run_mrc.py 전략 세팅

run_mrc를 사용하기 위해서 참고해야 하는 Args들을 알아봅시다.

```json
{
    "alias": "RUN_MRC_BASE_CONFIG",
    "model": {
        "model_name_or_path": "monologg/koelectra-small-v3-discriminator",
        "reader_name": "DPR",
    },
    "data": {
        "dataset_name": "train_dataset",
        "sub_datasets": "kor_dataset",
        "sub_datasets_ratio": "0.4",
        "overwrite_cache": false,
        "preprocessing_num_workers": 2,
        "max_seq_length": 384,
        "pad_to_max_length": false,
        "doc_stride": 128,
        "max_answer_length": 30,
    },
    "train": {
        "save_total_limit": 2,
        "save_steps": 500,
        "logging_steps": 100,
        "overwrite_output_dir": true,
        "do_train": true,
        "do_eval": true,
        "report_to": ["wandb"]
    },
    "retriever": {
        "retrain": false,
    }
}
```

- model
    - model_name_or_path: backbone 모델을 선택합니다.
    - reader_name: READER 모델을 선택합니다.
    > DPR 구조의 Reader모델에 backbone 모델로는 koelectra를 선택한 것

- data
    - dataset_name: 학습으로 사용하는 DATASET입니다. Default는 `train_dataset`이고 `squad_kor_v1`을 선택할 수 있습니다.
    - sub_datasets: 샘플링 된 데이터셋을 추가로 사용합니다. Default는 ''이고 빈 문자열일 경우에 추가하지 않습니다.
    - sub_datasets_ratio: 샘플링 된 데이터셋의 비율을 결정합니다.


### run_retrieval.py 전략 세팅


```json
{
    "alias": "RUN_RETRIEVAL_BASE_CONFIG",
    "model": {
        "retriever_name": "DPRBERT",
        "tokenizer_name": "xlm-roberta-large",
    },
    "data": {
        "dataset_name": "train_dataset",
        "sub_datasets": "",
    },
    "train": {
        "save_total_limit": 2,
    },
    "retriever": {
        "retrain": true,
        "dense_train_dataset": "train_dataset",
        "topk": 30, 
        "learning_rate": 3e-5,
        "per_device_train_batch_size": 16, 
        "per_device_eval_batch_size": 4,
        "num_train_epochs": 15, 
        "weight_decay": 0.01
    }  
}
```

- model
    - retriever_name: DPR RETRIEVER를 사용합니다. `backbone`모델은 기본적으로 `다국어 bert`를 사용하고 있습니다.
    - tokenizer_name: BM25 RETRIEVER에서 사용하는 tokenizer를 설정합니다. `xlm-roberta-large`가 현재는 최고 성능입니다.
- data
    - dataset_name: RETRIEVER성능 평가를 하기위해서 KLUE MRC 데이터셋을 사용합니다. validation 부분을 사용합니다.
    - sub_datasets: "", 빈 문자열로 설정해줍니다, 사실 추가되도 validation만 사용해서 큰 문제는 없습니다.
- retriever
    - retrain: true일 경우 재 학습을 진행합니다.
        - 경로: `../input/embed/{retriever_name}`
        - 경로에 파일이 존재할 경우 재학습을 진행하지 않지만 true일 경우 파일을 덮어씌웁니다.
    - dense_train_dataset: dense retriever같은 경우 새로운 데이터 셋이 몇개 추가되어서, 학습에 사용할 데이터셋을 추가 할 수 있습니다.
        - train_dataset: 기본적인 데이터 셋이며, batch_size를 16까지 설정할 수 있습니다.
        - bm25_document_questions, bm25_question_documents: BM25로 구축된 데이터 셋이며, batch_size를 무조건 1로 설정해야 합니다.
            - 여기서의 batch_size는 gradient_accumulation_step으로 조정하신다고 보시면 됩니다. ( 추가 예정 )
