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
        "load_best_model_at_end": true,
        "save_total_limit": 2,
        "save_strategy": "steps",
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

```
python -m run --strategis ST01,ST02 --run_cnt 3
```

ST01, ST02 전략을 다른 seed값으로 3번씩 실행

```
python -m run --strategis ST01 --run_cnt 3
```

단일 실행도 가능합니다.
