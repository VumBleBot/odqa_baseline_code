![vumblebot-logo](https://i.imgur.com/ch7hFii.png)

<b>Open-Domain Question Answering(ODQA)</b>는 다양한 주제에 대한 문서 집합으로부터 자연어 질의에 대한 답변을 찾아오는 task입니다. 이때 사용자 질의에 답변하기 위해 주어지는 지문이 따로 존재하지 않습니다. 따라서 사전에 구축되어있는 Knowledge Resource(본 글에서는 한국어 Wikipedia)에서 질문에 대답할 수 있는 문서를 찾는 과정이 필요합니다.

**VumBleBot**은 ODQA 문제를 해결하기 위해 설계되었습니다. 질문에 관련된 문서를 찾아주는 Retriever, 관련된 문서를 읽고 간결한 답변을 내보내주는 Reader가 구현되어 있습니다. 이 두 단계를 거쳐 만들어진 VumBleBot은 어떤 어려운 질문을 던져도 척척 답변을 해주는 질의응답 시스템입니다.

[:bookmark_tabs: **Wrap-up report**](https://hackmd.io/@9NfvP9AZQL2Psilxs3oNBA/SyH-EkVt_)에 모델, 실험 관리 및 검증 전략, 앙상블, 코드 추상화 등 저희가 다룬 기술의 흐름과 고민의 흔적들이 담겨있습니다.

# VumbleBot - BaselineCode  <!-- omit in toc -->

- [DEMO](#demo)
  - [Reader](#reader)
  - [Retrieval](#retrieval)
- [TIPS](#tips)
- [Simple Use](#simple-use)
  - [predict](#predict)
  - [reader train/validation](#reader-trainvalidation)
  - [retriever train/validation](#retriever-trainvalidation)
  - [reader, retriever validation](#reader-retriever-validation)
  - [make dataset](#make-dataset)
- [File Structure](#file-structure)
  - [input](#input)
  - [baseline_code](#baseline_code)
- [Json File Example](#json-file-example)
- [Usage](#usage)
  - [Usage: Train](#usage-train)
    - [READER Train](#reader-train)
    - [READER Result](#reader-result)
    - [RETRIEVER Train](#retriever-train)
    - [RETRIEVER Result](#retriever-result)
  - [Usage: Predict](#usage-predict)
    - [Predict result](#predict-result)
- [TDD](#tdd)

## DEMO

### Reader

```
python -m run_mrc --strategies RED_DPR_BERT --run_cnt 1 --debug False --report False
```

![image](https://user-images.githubusercontent.com/40788624/119266204-cf6ffe80-bc24-11eb-9d33-369c239b857e.png)

### Retrieval

```
python -m run_retrieval --strategies RET_05_BM25_DPRBERT,RET_06_TFIDF_DPRBERT,RET_07_ATIREBM25_DPRBERT --run_cnt 1 --debug False --report False
```

![retriever-top-k-compare](https://user-images.githubusercontent.com/40788624/119266107-6daf9480-bc24-11eb-85f5-6f6f09691c9b.png)

## TIPS

- [전체적인 내용](./documents/README.md)
- [READER class](./documents/reader.md)
- [RETRIEVER class](./documents/retriever.md)

## Simple Use

### predict
```bash
python -m predict --strategies ST01 
```

### reader train/validation

```bash
python -m run_mrc --strategies ST01,ST02 --debug True --report False --run_cnt 1
python -m run_mrc --strategies ST01,ST02 --debug False --report True --run_cnt 3
```

### retriever train/validation

```bash
python -m run_retrieval --strategies ST01,ST02 --debug True --report False --run_cnt 1
python -m run_retrieval --strategies ST01,ST02 --debug False --report True --run_cnt 3
```

### reader, retriever validation

```bash
python -m run --strategies ST01,ST02 --debug True --report False --run_cnt 1
python -m run --strategies ST01,ST02 --debug False --report True --run_cnt 3
```

### make dataset

```bash
python -m make_dataset.cheat_dataset
python -m make_dataset.kor_sample_dataset
python -m make_dataset.qd_pair_bm25
```

## File Structure  

### input
  
```
input/
│ 
├── config/ - strategies
│   ├── ST01.json
│   └── ...
│
├── checkpoint/ - checkpoints&predictions (strategy_alias_seed)
│   ├── ST01_base_00
│   │   ├── checkpoint-500
│   │   └── ...
│   ├── ST01_base_95
│   └── ...
│ 
├── data/ - competition data
│   ├── dummy_data/
│   ├── train_data/
│   └── test_data/
│
├─── embed/ - embedding caches of wikidocs.json
│   ├── TFIDF
│   │   ├── TFIDF.bin
│   │   └── embedding.bin
│   ├── BM25
│   │   ├── BM25.bin
│   │   └── embedding.bin
│   ├── ATIREBM25
│   │   ├── ATIREBM25.bin
│   │   ├── ATIREBM25_idf.bin
│   │   └── embedding.bin
│   ├── DPRBERT
│   │   ├── DPRBERT.pth
│   │   └── embedding.bin
│   └── ATIREBM25_DPRBERT
│       └── classifier.bin
│
└── keys/ - secret keys or tokens
    └── secrets.json
```
    
### baseline_code
  
```
odqa_baseline_code/
│
├── config/ - arguments config file
│   ├── README.md
│   ├── model_args.py
│   ├── data_args.py
│   ├── retriever_args.py
│   └── train_args.py
│
├── reader/ - reader 
│   ├── base_reader.py
│   ├── custom_head.py
│   ├── custom_reader.py
│   └── pororo_reader.py
│
├── retrieval/ - retriever
│   ├── base_retrieval.py
│   ├── dense
│   │   ├── dense_base.py
│   │   ├── dpr.py
│   │   ├── dpr_base.py
│   │   └── colbert.py
│   ├── hybrid
│   │   ├── hybrid_base.py
│   │   └── hybrid.py
│   └── sparse
│       ├── sparse_base.py
│       ├── tfidf.py
│       ├── bm25.py
│       ├── atire_bm25.py
│       └── ...
│
├── make_dataset/ - make necessary datasets
│   ├── aggregate_wiki.py
│   ├── kor_sample_dataset.py
│   ├── negative_ctxs_dataset.py
│   ├── qd_pair_bm25.py
│   ├── triplet_dataset.py
│   └── ...
│ 
├── utils/ - utils
│   ├── evaluation.py - for evaluation normalize
│   ├── prepare.py - get datasets/retriever/reader
│   ├── slack_api.py - for slack api loading, report to slack channel
│   ├── tokenization_kobert.py - for kobert tokenizer
│   ├── tools.py - update arguments, tester excuter
│   ├── tester.py - debugging, testing
│   ├── trainer_qa.py - trainer(custom evaluate, predict)
│   └── utils_qa.py - post processing function
│
├── ensemble.py - do ensemble
├── run_mrc.py - train/evaluate MRC model
├── run_retriever.py - train/evaluate retriever model
├── run.py - evaluate both models
└── predict.py - inference
```

## Json File Example

ST00.json 하이퍼파라미터는 아래 파일들을 참고해서 수정할 수 있습니다.

- config/model_args.py
- config/train_args.py
- config/data_args.py
- config/retriever_args.py
- config/readme.md

```json
{
    "alias": "vumblebot",
    "model": {
        "model_name_or_path": "monologg/koelectra-small-v3-discriminator",
        "retriever_name": "BM25_DPRKOBERT",
        "reader_name": "CNN",
        "config_name": "",
        "tokenizer_name": ""
    },
    "data": {
        "dataset_name": "train_dataset",
        "sub_datasets": "kor_dataset,etr_dataset",
        "sub_datasets_ratio": "0.2,0.3", 
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
    },
    "retriever": {
        "b": 0.01,
        "k1": 0.1,
        "topk": 5,
        "alpha": 0.1,
        "retrain": false,
        "weight_decay": 0.01,
        "learning_rate": 3e-5,
        "num_train_epochs": 2,
        "per_device_eval_batch_size": 2,
        "gradient_accumulation_steps": 1,
        "per_device_train_batch_size": 4,
        "dense_train_dataset": "train_dataset"
    }
}
```

## Usage

Server의 디렉토리 구조에서 input과 같은 수준에 위치하면 됩니다.

```
root/  
├── input/
└── odqa_baseline_code/  
```

### Usage: Train   
  
#### READER Train

- ST01 전략을 서로 다른 seed 값으로 3번 실행  
`python -m run --strategies ST01 --run_cnt 3`
- ST01, ST02 전략을 서로 다른 seed 값으로 3번씩 실행 (총 6번)   
`python -m run --strategies ST01,ST02 --run_cnt 3`

#### READER Result  

```
input/  
└── checkpoint/  
    ├── ST02_temp_95/
    │   ├── checkpoint-500/
    │   └── ...
    ├── nbest_predictions_valid.json
    └── predictions_valid.json
```

#### RETRIEVER Train

- 전략(ST00.json)에 있는 Reader 모델은 사용하지 않습니다.
- Retriever 모델은 학습이 완료된 이후로는 결과가 불변이기 때문에 run_cnt 값을 1로 설정해주시면 됩니다.
- retrain 인자를 사용해서 재학습을 진행할 수 있습니다.

`python -m run_retrieval --strategies ST07,ST08,ST09 --run_cnt 1`

#### RETRIEVER Result

- wandb.ai에서 이미지를 확인 할 수 있습니다.

```
전략: RET_07_ATIREBM25_DPRBERT: ATIREBM25_DPRBERT
TOPK: 1 ACC: 75.00
TOPK: 2 ACC: 85.83
TOPK: 3 ACC: 90.42
TOPK: 4 ACC: 90.83
TOPK: 5 ACC: 91.67
TOPK: 6 ACC: 93.33
TOPK: 7 ACC: 95.00
TOPK: 8 ACC: 95.42
TOPK: 9 ACC: 95.83
TOPK: 10 ACC: 96.25
```

![image](https://user-images.githubusercontent.com/40788624/119265923-9c793b00-bc23-11eb-8439-c237fa91f6bb.png)

```
input
└── embed
    ├── TFIDF
    │   ├── TFIDF.bin
    │   └── embedding.bin
    ├── BM25
    │   ├── BM25.bin
    │   └── embedding.bin
    ├── ATIREBM25
    │   ├── ATIREBM25.bin
    │   ├── ATIREBM25_idf.bin
    │   └── embedding.bin
    ├── DPRBERT
    │   ├── DPRBERT.pth
    │   └── embedding.bin
    └── ATIREBM25_DPRBERT
        └── classifier.bin
```


### Usage: Predict

- strategies로 한 개의 전략만 집어넣는 것을 추천합니다.  
`python -m run --strategies ST01`
  
#### Predict result  
 
```
input/  
└── checkpoint/  
    └── ST01/
        ├── nbest_predictions_test.json
        ├── predictions_test.json
        ├── valid_results.json
        └── (optional) pororo_predictions_test.json
```
  
단일 실행도 가능합니다.
  
## TDD
| [tester.py](./utils/tester.py) : 구현된 기능이 정상 작동되는지 테스트     

- 검증할 전략을 옵션으로 입력  

    `python -m utils.tester --strategies ST02,ST01`  
    `python -m run --strategies ST01`  


- (example) 결과 해석
 
    - 5가지 단위 테스트 중 1 fail, 1 error 발생     
    ```

    ===================================================
    ERROR: test_strategies_with_dataset (__main__.TestReader)
    (Constraint)
    ---------------------------------------------------
    .... 

    ===================================================
    FAIL: test_valid_dataset (__main__.TestReader)
    ---------------------------------------------------
    ....

    Traceback (most recent call last):
    File "/opt/ml/odqa_baseline_code/tester.py", line 42, in test_valid_dataset
        assert False, "존재하지 않는 dataset입니다. "
    AssertionError: 존재하지 않는 dataset입니다. 

    ---------------------------------------------------
    Ran 5 tests in 11.046s

    FAILED (failures=1, errors=1)
    ```
    - 5가지 단위 테스트 모두 통과 
    ```
    ----------------------------
    Ran 5 tests in 76.858s

    OK
    ```
