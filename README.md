# VumbleBot - BaselineCode <!-- omit in toc -->

- [Branch](#branch)
- [File Structure](#file-structure)
  - [input](#input)
  - [baseline_code](#baseline_code)
- [Json File Example](#json-file-example)
- [Usage](#usage)
  - [Usage: Train](#usage-train)
    - [Train result](#train-result)
  - [Usage: Predict](#usage-predict)
    - [Predict result](#predict-result)
- [TDD](#tdd)

## Branch 

- 코드 수정 후 main branch로 pull request

## File Structure

**굵게** 칠해진 곳은 아직 코딩중입니다.. :(

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
└── config/ - arguments
    ├── data_args.py
    ├── model_args.py
    └── train_args.py
```
    
### baseline_code
  
```
odqa_baseline_code/
│
├── reader/ - reader
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
- config/readme.md

```json
{
    "alias": "base",
    "model": {
        "model_name_or_path": "monologg/koelectra-small-v3-discriminator",
        "config_name": "",
        "tokenizer_name": ""
    },
    "data": {
        "dataset_name": "train_dataset",
        "overwrite_cache": false,
        "preprocessing_num_workers": 4,
        "max_seq_length": 384,
        "pad_to_max_length": false,
        "doc_stride": 128,
        "max_answer_length": 30,
        "train_retrieval": true,
        "eval_retrieval": true
    },
    "train": {
        "do_train": true,
        "do_eval": true,
        "save_total_limit": 2,
        "save_steps": 100,
        "logging_steps": 100,
        "overwrite_output_dir": true,
        "report_to": ["wandb"]
    }
}
```

## Usage

Server의 디렉토리 구조에서 input과 같은 수준에 위치하면 됩니다.

- input
- code
- new_baseline_code

### Usage: Train   
  
- ST01 전략을 서로 다른 seed 값으로 3번 실행  
`python -m run --strategies ST01 --run_cnt 3`    
- ST01, ST02 전략을 서로 다른 seed 값으로 3번씩 실행 (총 6번)   
`python -m run --strategies ST01,ST02 --run_cnt 3`   
  
#### Train result  
  
- input
    - checkpoint
        -ST02_95_temp
            -checkpoint...
        - nbest_predictions_valid.json
        - predictions_valid.json

### Usage: Predict

- strategies로 한 개의 전략만 집어넣는 것을 추천합니다.  
`python -m run --strategies ST01 --model_path ../input/checkpoint/ST02_95_temp/checkpoint-500`  
  
#### Predict result  
 
- input
    - checkpoint
        -ST01
            - nbest_predictions_test.json
            - predictions_test.json


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
