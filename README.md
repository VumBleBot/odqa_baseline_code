# VumbleBot - BaselineCode

## Branch 

- 코드 수정 후 main branch로 pull request

## ISSUE

- run.py에서 train 이후에 evaluate 진행
- inference.py -> predict.py로 파일 수정

## 파일 구조

**굵게** 칠해진 곳은 아직 코딩중입니다.. :(

- input
    - checkpoint (strategy, seed, alias: 별칭)
        - ST01_95_base
            - ...
        - ST02_95_temp
            - ...
    - config
        - ST01.json
        - ST02.json
    - data (Competiton 데이터)
        - train_data
        - test_data
        - dummy_data
    - embed (데이터랑 알고리즘에 종속적인데 데이터(wikidocs.json)는 변하지 않음)
        - TFIDF
            - embeddding.bin
            - tfidv.bin
        - BM25
            - embeddding.bin
            - bm25.bin
        - desne_bert ( dense는 docs임베딩을 저장하거나, token vector를 저장할 듯 싶습니다.)
            - embeddding.bin
            - dense_bert.bin
        ...
    - **info (시각화에 사용되기 위한 정보들 Logging)**
        - ST01_95_base.json
        - ST02_95_temp.json
- new_baseline_code
    - post_process
        - answer
        - document
    - retriever
        - sparse
        - dense
    - predict.py
    - prepare.py 
    - run.py 
    - tokenization_bert.py ( `kobert`, `distilkobert` )
    - tools.py
    - trainer_qa.py
    - utils_qa.py
    - tester.py


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
        "save_total_limit": 2,
        "save_steps": 100,
        "logging_steps": 100,
        "overwrite_output_dir": true,
        "report_to": ["wandb"]
    }
}
```

## How to Usage

Server의 디렉토리 구조에서 input과 같은 수준에 위치하면 됩니다.

- input
- code
- new_baseline_code

### How to Usage: Train

```
python -m run --strategis ST01,ST02 --run_cnt 3
```

ST01, ST02 전략을 다른 seed값으로 3번씩 실행

```
python -m run --strategis ST01 --run_cnt 3
```

**Train Reulst**

- input
    - checkpoint
        -ST02_95_temp
            -checkpoint...
        - nbest_predictions_valid.json
        - predictions_valid.json

### How to Usage: Predict

- strategis에 한 개의 전략만 집어넣는 것을 추천합니다.

```
python -m run --strategis ST01 --model_path ../input/checkpoint/ST02_95_temp/checkpoint-500
```

**Predict Reulst**

- input
    - checkpoint
        -ST01
            - nbest_predictions_test.json
            - predictions_test.json


단일 실행도 가능합니다.

# TDD
| [tester.py](./tester.py) : 구현된 기능이 정상 작동되는지 테스트     

검증할 전략을 옵션으로 입력

```
python -m tester --strategis ST02,ST01
```

```
python -m run --strategis ST01
```

- [example] 결과 해석
 
    - 5가지 단위 테스트 중 1 fail, 1 error 발생     
    ```

    ===================================================
    ERROR: test_strategis_with_dataset (__main__.TestReader)
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
