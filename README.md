# VumbleBot - BaselineCode <!-- omit in toc -->

- [TIPS](#tips)
- [Branch](#branch)
- [Simple Use](#simple-use)
  - [predict](#predict)
  - [reader train/validation](#reader-trainvalidation)
  - [retriver train/validation](#retriver-trainvalidation)
  - [reader, retriver validation](#reader-retriver-validation)
  - [make dataset](#make-dataset)
- [File Structure](#file-structure)
  - [input](#input)
  - [baseline_code](#baseline_code)
- [Json File Example](#json-file-example)
- [Usage](#usage)
  - [Usage: Train](#usage-train)
    - [READER Train](#reader-train)
    - [READER Result](#reader-result)
    - [RETRIVER Train](#retriver-train)
    - [RETRIVER Result](#retriver-result)
  - [Usage: Predict](#usage-predict)
    - [Predict result](#predict-result)
- [TDD](#tdd)


## TIPS

- [전체적인 내용](./documents/README.md)
- [READER class](./documents/reader.md)
- [RETRIVER class](./documents/retriever.md)


## Branch 

- 코드 수정 후 main branch로 pull request

## Simple Use

> 학습 후 메모리 해제가 완벽하게 안 되는 이슈가 있습니다! 한 번에 너무 많이 돌리는 것만 지양하면 괜찮을 것 같습니다!

### predict

**지금 방식은 비효율적이라서 곧 수정될 예정입니다!**
    - 한 번에 하나의 전략만!
    - model_path와 전략을 맞춰야 됩니다!
    
```bash
python -m predict --strategies ST01 --model_path {MODEL_PATH}
python -m predict --strategies ST01 --model_path ../input/checkpoint/ST01_base_95/checkpoint-1100
```

### reader train/validation

```bash
python -m run_mrc --strategies ST01,ST02 --debug True --report False --run_cnt 1
python -m run_mrc --strategies ST01,ST02 --debug False --report True --run_cnt 3
```

### retriver train/validation

```bash
python -m run_retrieval --strategies ST01,ST02 --debug True --report False --run_cnt 1
python -m run_retrieval --strategies ST01,ST02 --debug False --report True --run_cnt 3
```

### reader, retriver validation

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
│   ├── ST01_base_95
│   └── ...
│ 
├── data/ - competition data
│   ├── dummy_data/
│   ├── train_data/
│   └── test_data/
│
├── embed/ - embedding caches of `wikidocs.json`
│   ├── TFIDF/
│   │   ├── embedding.bin
│   │   └── tfidv.bin
│   ├── BM25/
│   └── DPR/
│
├── info/ - logging (for visualization)
│   └── NOT IMPLEMENTED YET
│
├── config/ - arguments
│    ├── data_args.py
│    ├── model_args.py
│    └── train_args.py
│ 
└── keys/ - secret keys or tokens
    └── secrets.json
```
    
### baseline_code
  
```
odqa_baseline_code/
│
├── reader/ - reader
│   ├── pororo_reader.py
│   └── base_reader.py
│
├── retrieval/ - retriever
│   ├── sparse/
│   │   ├── tfidf.py
│   │   └── bm25.py
│   │
│   └── dense/
│       └── dpr.py
│       
├── trainer_qa.py - trainer(custom evaluate, predict)
├── utils_qa.py - post processing function
├── prepare.py - get datasets/retriever/reader    
├── tools.py - arguments/tester
│
├── tester.py - debugging, testing
│
├── run.py - train/evaluate
├── predict.py - inference
│
├── tokenization_kobert.py - tokenizer ( `kobert`, `distilkobert` )
│
│
└── config/ - arguments
    ├── data_args.py
    ├── model_args.py
    └── train_args.py
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
    "alias": "base",
    "model": {
        "model_name_or_path": "monologg/koelectra-small-v3-discriminator",
        "model_path": "",
        "retriever_name": "BM25_DPRKOBERT",
        "reader_name": "DPR",
        "config_name": "",
        "tokenizer_name": ""
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
        "train_retrieval": true,
        "eval_retrieval": true
    },
    "train": {
        "masking_ratio": 0.05,
        "do_train": true,
        "do_eval": true,
        "pororo_prediction": true,
        "save_total_limit": 2,
        "save_steps": 100,
        "logging_steps": 100,
        "overwrite_output_dir": true,
        "report_to": ["wandb"]
    },
    "retriever": {
        "b": 0.01,
        "k1": 0.1,
        "topk": 30,
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
├── code/  
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

#### RETRIVER Train

- 전략(ST00.json)에 있는 Reader 모델은 사용하지 않습니다.
- Retriver 모델은 학습이 완료된 이후로는 결과가 불변이기 때문에 run_cnt 값을 1로 설정해주시면 됩니다.
- retrain 인자를 사용해서 재학습을 진행할 수 있습니다.

`python -m run_retrieval --strategies ST01,ST02,ST03,ST04 --run_cnt 1`

#### RETRIVER Result

- wandb.ai에서 이미지를 확인 할 수 있습니다.

```
전략: ST01 RETRIEVER: TFIDF
TOPK: 0 ACC: 22.08
TOPK: 1 ACC: 32.92
TOPK: 2 ACC: 36.25
TOPK: 3 ACC: 40.42
TOPK: 4 ACC: 43.33
TOPK: 5 ACC: 47.08
TOPK: 6 ACC: 49.58
TOPK: 7 ACC: 51.25
TOPK: 8 ACC: 52.92
TOPK: 9 ACC: 55.00
```

![image](https://user-images.githubusercontent.com/40788624/117123189-b0264400-add1-11eb-8ec6-77b05097d4c8.png)

```
input
└── embed
    ├── BM25
    │   ├── BM25.bin
    │   └── embedding.bin
    └── TFIDF
        ├── TFIDF.bin
        └── embedding.bin
```
 

### Usage: Predict

- strategies로 한 개의 전략만 집어넣는 것을 추천합니다.  
`python -m run --strategies ST01 --model_path ../input/checkpoint/ST02_95_temp/checkpoint-500`  
  
#### Predict result  
 
```
input/  
└── checkpoint/  
    └── ST01/
        ├── nbest_predictions_test.json
        └── predictions_test.json
```
  
단일 실행도 가능합니다.
  
## TDD
| [tester.py](./tester.py) : 구현된 기능이 정상 작동되는지 테스트     

- 검증할 전략을 옵션으로 입력  

    `python -m tester --strategies ST02,ST01`  
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
