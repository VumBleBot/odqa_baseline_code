# ODQA HELPER DOCUMENT

- [reader.md](reader.md): reader 구조에 대해 설명되어 있습니다.
- [retriever.md](retriever.md): retriever 구조에 대해 설명되어 있습니다.


## 전체적인 구조

모듈 기준으로 프로젝트의 전체적인 구조를 나열하였습니다.

### 실행 모듈

구현한 모듈들을 실제로 실행하는데에 관여하는 메인 실행 모듈입니다.

- `run.py` : RETRIVER와 READER모델을 함께 활용하여 종합적으로 ODQA 성능을 평가하는 모듈입니다.
    - `run_mrc.py`와 관련있는 모듈들
    - `run_retrieval.py`와 관련있는 모듈들
- `run_mrc.py` : 문서 검색 없이 READER의 기계 독해 능력을 평가하는 모듈입니다.
    - `reader/*`
    - `prepare.py`
        - get_dataset: 데이터 셋을 가져옵니다.
        - get_reader: READER 모델을 불러옵니다.
- `run_retrieval.py` : RETRIEVER의 검색 성능을 평가하는 모듈입니다.
    - `retriever/*`
    - `prepare.py`
        - get_dataset: 데이터 셋을 가져옵니다.
        - get_retriever: RETRIEVER 모델을 불러옵니다.
- `predict.py` : test\_dataset에 대한 결과값을 예측하는 모듈입니다.
    - `prepare.py`
        - get_dataset: 
        - get_reader: READER 모델을 불러옵니다.
        - get_retriever: RETRIEVER 모델을 불러옵니다.


### run.py 전략 세팅

- `run.py`는 `run_mrc.py`와 `run_retrieval.py`에 사용하는 세팅을 함께 활용합니다.

### run_mrc.py 전략 세팅

run_mrc를 사용하기 위해 필요한 argument들을 알아봅니다.

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
        "masking_ratio": 0.0,
        "do_train": true,
        "do_eval": true,
        "do_eval_during_training": true,
        "eval_step": 500,
        "pororo_prediction": false,
        "save_total_limit": 5,
        "save_steps": 100,
        "logging_steps": 100,
        "overwrite_output_dir": true,
        "freeze_backbone": false,
        "report_to": ["wandb"]
    }
}
```

- model
    - `model_name_or_path`: backbone 모델을 선택합니다.
    - `reader_name`: READER 모델을 선택합니다.
    > DPR 구조(fully-connected head)의 Reader모델에 backbone 모델로는 koelectra를 선택한 것

- data
    - `dataset_name`: 학습에 사용하는 데이터셋입니다. "squad_kor_v1"이면 KorQuAD로, "train_dataset"이면 `input/data`에 존재하는 custom dataset으로 학습을 진행합니다. 
    - `sub_datasets`: 추가적인 데이터셋을 샘플링하여 학습에 활용합니다. 빈 문자열("")이면 추가 데이터셋을 활용하지 않습니다.
    - `sub_datasets_ratio`: 추가 데이터셋의 비율을 결정합니다. sub_datasets이 빈 문자열이면 무시됩니다.

- train
    - `masking_ratio`: 학습 단계에서 masking을 적용하려면 
    - `do_eval_during_training`: 학습 단계에서 매 `eval_step`마다 evaluation을 진행합니다.
    - `pororo_prediction`: validation 단계에서 `pororo`로 보완된 결과값을 저장할지 결정합니다. (`pororo`가 무조건적인 성능 향상을 보장하지는 않습니다!)
    - `freeze_backbone`: backbone을 얼리고 head만 학습을 진행할지 결정합니다.

### run_retrieval.py 전략 세팅

```json
{
    "alias": "RUN_RETRIEVAL_BASE_CONFIG",
    "model": {
        "retriever_name": "DPRBERT",
        "tokenizer_name": "xlm-roberta-large",
    },
    "data": {
        "dataset_name": "train_dataset"
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
    - `retriever_name`: DPR RETRIEVER를 사용합니다. `backbone`모델은 기본적으로 다국어 BERT 모델(bert-multilingual)을 사용하고 있습니다.
    - `tokenizer_name`: BM25 RETRIEVER에서 사용하는 tokenizer를 설정합니다. 프로젝트 내 실험에서는 `xlm-roberta-large` tokenizer가 최고 성능을 보였습니다.
- data
    - `dataset_name`: retriever 성능 평가를 위해 활용할 데이터셋입니다.
- retriever
    - `retrain`: `true`일 경우 retriever의 embedding을 재학습하고 기존 파일에 덮어씌웁니다.
        - 경로: `../input/embed/{retriever_name}`
        - 경로에 파일이 존재하지 않으면 이 argument 값과 무관하게 embedding을 학습 및 저장합니다.
    - `dense_train_dataset`: dense retriever의 경우 학습에 사용할 데이터셋을 추가할 수 있습니다.
        - train_dataset: custom dataset으로 batch_size를 16까지 설정할 수 있습니다.
        - bm25_document_questions, bm25_question_documents: BM25로 구축된 데이터 셋이며, batch_size를 반드시 1로 설정해야 합니다.
            - 여기서의 batch_size는 gradient_accumulation_step으로 조정할 수 있습니다.
