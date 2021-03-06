![vumblebot](https://user-images.githubusercontent.com/61135159/122643559-660fdc80-d14b-11eb-8595-a12589a7b77e.png)

<b>Open-Domain Question Answering(ODQA)</b>는 다양한 주제에 대한 문서 집합으로부터 자연어 질의에 대한 답변을 찾아오는 task입니다. 이때 사용자 질의에 답변하기 위해 주어지는 지문이 따로 존재하지 않습니다. 따라서 사전에 구축되어있는 Knowledge Resource(본 글에서는 한국어 Wikipedia)에서 질문에 대답할 수 있는 문서를 찾는 과정이 필요합니다.

**VumBleBot**은 ODQA 문제를 해결하기 위해 설계되었습니다. 질문에 관련된 문서를 찾아주는 Retriever, 관련된 문서를 읽고 간결한 답변을 내보내주는 Reader가 구현되어 있습니다. 이 두 단계를 거쳐 만들어진 VumBleBot은 어떤 어려운 질문을 던져도 척척 답변을 해주는 질의응답 시스템입니다.

[:bookmark_tabs: **Wrap-up report**](https://hackmd.io/@9NfvP9AZQL2Psilxs3oNBA/SyH-EkVt_)에 모델, 실험 관리 및 검증 전략, 앙상블, 코드 추상화 등 저희가 다룬 기술의 흐름과 고민의 흔적들이 담겨있습니다.

# VumBleBot - BaselineCode  <!-- omit in toc -->

- [Tutorial](#tutorial)
- [DEMO](#demo)
  - [Reader](#reader)
  - [Retrieval](#retrieval)
- [Installation](#installation)
  - [Dependencies](#dependencies)
- [File Structure](#file-structure)
  - [Baseline code](#baseline-code)
  - [Input](#input)
- [Json File Example](#json-file-example)
- [Dataset setting](#dataset-setting)
- [Usage](#usage)
  - [Usage: Directory setting](#usage-directory-setting)
  - [Usage: Train](#usage-train)
    - [Reader](#reader-1)
    - [Retriever](#retriever)
  - [Usage: Validation](#usage-validation)
  - [Usage: Predict](#usage-predict)
  - [Usage: Make additional dataset](#usage-make-additional-dataset)
- [Test Driven Development](#test-driven-development)
- [Contributors](#contributors)
- [Reference](#reference)
  - [Papers](#papers)
  - [Dataset](#dataset)
- [License](#license)

## Tutorial  

Google Colab환경에서 간편하게 VumBleBot의 베이스라인 코드를 사용할 수 있습니다.
<table>
  <tr><td><strong>ODQA Tutorial</strong></td><td><a href="https://colab.research.google.com/drive/1r1wJ4emb-esyq1L-R2_Aix81k44lJtwK?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg"/></a></td></tr>
  <tr><td><strong>Retrieval Tutorial</strong></td><td><a href="https://colab.research.google.com/drive/1U1P_WoNbBV3XO7I_YqVJmggPKVtQUXgi?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg"/></a></td></tr>
  <tr><td><strong>Reader Tutorial</strong></td><td><a href="https://colab.research.google.com/drive/1EZZy3xkSM1XbC9M9X1aeVpvMa7iKPP0W?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg"/></a></td></tr>
</table>
   
## DEMO

아래 문서에서 사용할 수 있는 reader/retriever 모델을 확인하실 수 있습니다.  

- [Overall](./documents/README.md)
- [Reader class](./documents/reader.md)
- [Retriever class](./documents/retriever.md)

### Reader

```
python -m run_mrc --strategies RED_DPR_BERT --run_cnt 1 --debug False --report True
```

![image](https://user-images.githubusercontent.com/40788624/120093538-f3b46980-c155-11eb-938e-f8b44197d01b.png)

### Retrieval

```
python -m run_retrieval --strategies RET_05_BM25_DPRBERT,RET_06_TFIDF_DPRBERT,RET_07_ATIREBM25_DPRBERT \
                        --run_cnt 1 \
                        --debug False \
                        --report False
```

![retriever-top-k-compare](https://user-images.githubusercontent.com/40788624/119266107-6daf9480-bc24-11eb-85f5-6f6f09691c9b.png)  
  
## Installation
### Dependencies
- fuzzywuzzy==0.18.0
- konlpy==0.5.2
- numpy==1.19.4 
- pandas==1.1.4 
- pororo==0.4.2 
- scikit-learn==0.24.1  
- sentencepiece==0.1.95 
- slack-sdk==3.5.1 
- torch==1.7.1 
- tqdm==4.41.1 
- transformers==4.5.1  
- datasets==1.5.0
- wandb==0.10.27 

```
pip install -r requirements.txt
```

:exclamation: **이 프로젝트는 `mecab`을 사용합니다.**  
[KoNLPy 공식 홈페이지](https://konlpy.org/en/latest/install/)를 참고하여 KoNLPy 및 MeCab 설치를 진행해주세요.  

:exclamation: **현재 CUDA 버전이 낮을 경우 pororo 설치 시 GPU가 활성화되지 않는 이슈가 존재합니다.**  
만약 `pororo` 설치 이후 학습속도가 지나치게 느려졌을 경우, 아래 명령으로 `torch` 버전업을 통해 GPU를 활성화해주세요. [이슈 참고](https://github.com/VumBleBot/odqa_baseline_code/issues/82)

```
pip install torch==1.7.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

## File Structure  
### Baseline code
  
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
│       ├── bm25_base.py
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
├── examples/ - strategy files
│   ├── ST01.json
│   └── ...
│
├── scripts/ - executable script files
│   ├── run_mrc.sh - execute run_mrc module
│   ├── run_retrieval.sh - execute run_retrieval module
│   ├── run.sh - execute run module
│   └── predict.sh - execute predict module
│
├── ensemble.py - do ensemble
├── run_mrc.py - train/evaluate MRC model
├── run_retrieval.py - train/evaluate retriever model
├── run.py - evaluate both models
└── predict.py - inference
```

### Input
  
```
input/
│
├── checkpoint/ - checkpoints&predictions (strategy_alias_seed)
│   ├── ST01_base_00
│   │   ├── checkpoint-500
│   │   └── ...
│   ├── ST01_base_95
│   └── ...
│ 
├── data/ - competition data
│   ├── wikipedia_documents.json
│   └── custom datasets(train_data/test_data) ...
│
├── embed/ - embedding caches of wikidocs.json
│   ├── TFIDF
│   │   ├── TFIDF.bin
│   │   └── embedding.bin
│   ├── BM25
│   │   ├── BM25.bin
│   │   └── embedding.bin
│   ├── ATIREBM25
│   │   ├── ATIREBM25_idf.bin
│   │   ├── ATIREBM25.bin
│   │   ├── embedding.bin
│   │   └── idf.bin
│   ├── DPRBERT
│   │   ├── DPRBERT.pth
│   │   └── embedding.bin
│   └── ATIREBM25_DPRBERT
│       └── classifier.bin
│
└── (optional) keys/ - secret keys or tokens
    └── (optional) secrets.json
```

## Json File Example

전략 config 파일(ST00.json) 예시입니다.    
arguments(hyperparameter)는 아래 파일들을 참고하여 수정하시면 됩니다.    

- [config/model_args.py](./config/model_args.py)
- [config/train_args.py](./config/train_args.py)
- [config/data_args.py](./config/data_args.py)
- [config/retriever_args.py](./config/retriever_args.py)
- [config/README.md](./config/README.md)

예시 전략 파일들이 [examples/](./examples)에 존재하니, 참고하셔서 전략 파일을 작성하시면 됩니다.  
  
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
        "dataset_name": "squad_kor_v1",
        "sub_datasets": "",
        "sub_datasets_ratio": "", 
        "overwrite_cache": false,
        "preprocessing_num_workers": 2,
        "max_seq_length": 384,
        "pad_to_max_length": false,
        "doc_stride": 128,
        "max_answer_length": 30
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
        "dense_train_dataset": "squad_kor_v1"
    }
}
```

## Dataset setting
본 프로젝트는 `transformers` 라이브러리를 통해 KorQuAD 1.0을 불러와 학습 및 검증을 수행합니다.    
**만약 custom dataset을 통해 학습을 수행하려면 아래와 같이 `input/data` 경로에 커스텀 데이터셋을 넣어주어야 합니다.**

```
input/
│
└── data
    ├── train_dataset
    │   ├── dataset_dict.json
    │   ├── train
    │   │   ├── dataset.arrow
    │   │   ├── dataset_info.json
    │   │   ├── indices.arrow
    │   │   └── state.json
    │   └── validation
    │       ├── dataset.arrow
    │       ├── dataset_info.json
    │       ├── indices.arrow
    │       └── state.json
    ├── test_dataset
    │   ├── dataset_dict.json
    │   └── validation
    │       ├── dataset.arrow
    │       ├── dataset_info.json
    │       ├── indices.arrow
    │       └── state.json
    └── wikipedia_documents.json
```

:exclamation: **predict를 수행하려면 `input/data/wikipedia_documents.json`과 `input/data/test_dataset`이 필수적으로 존재**해야합니다.  

- `wikipedia_documents.json`은 용량이 큰 관계로 프로젝트에서 직접적으로 제공하지 않습니다. [한국어 위키피디아](https://bit.ly/3yJ8KAl) 홈페이지에서 위키피디아 데이터를 다운받아 [examples/wikipedia_documents.json](./examples/wikipedia_documents.json)과 같은 형식으로 가공하여 활용하시면 됩니다.  
- `test_dataset`은 커스텀 데이터셋으로 [huggingface 공식 문서](https://huggingface.co/docs/datasets/v1.7.0/quicktour.html)를 참고하여 아래와 같은 형식으로 만들어 활용해주세요.  
  - Dataset 예시
    ```
    DatasetDict({
      validation: Dataset({
          features: ['id', 'question'],
          num_rows: 100
      })
    })
    ```

  - Data 예시
    ```
    {
      'id': '질문 ID(str)',
      'question': '질문(str)'
    }
    ```

- `train_dataset`은 KorQuAD로 모델 학습을 진행하실 경우 별도로 필요하지 않습니다. 커스텀 데이터셋으로 학습을 하려면 아래와 같은 형식으로 데이터셋을 만들어주세요.
  - Dataset 예시
    ```
    DatasetDict({
        train: Dataset({
            features: ['answers', 'context', 'document_id', 'id', 'question', 'title'],
            num_rows: 3000
        })
        validation: Dataset({
            features: ['answers', 'context', 'document_id', 'id', 'question', 'title'],
            num_rows: 500
        })
    })
    ```

  - Data 예시
    ```
    {
      'title': '제목(str)',
      'context': '내용(str)',
      'question': '질문(str)',
      'id': '질문 ID(str)',
      'answers': {'answer_start': [시작위치(int)], 'text': ['답(str)']},
      'document_id': 문서 ID(int)
    }
    ```

- 커스텀 데이터셋을 활용하여 **reader** 모델 학습을 하려면 [utils/prepare.py](./utils/prepare.py)를 참고하여 아래와 같이 전략 config를 수정해주세요.  
  ```
      ...
      "data": {
          "dataset_name": "train_dataset",
          "sub_datasets": "kor_dataset",
          "sub_datasets_ratio": "0.3", 
      ...
  ```
  - 커스텀 데이터셋을 활용하실 경우, KorQuAD 데이터셋을 위와 같이 `sub_datasets`로 주어 학습에 함께 활용할 수 있습니다. 이 때 `sub_datasets_ratio`를 이용하여 추가적인 데이터셋을 얼마나 활용할지 설정할 수 있습니다. 
  - `sub_datasets`를 활용하시려면 [아래 파트](#usage-make-additional-dataset)를 참고하여 추가적인 데이터셋을 생성해주세요.

- 커스텀 데이터셋을 활용하여 **dense retriever** 모델 학습을 하려면 아래와 같이 전략 config를 수정해주세요.
  ```
    ...
    "retriever": {
        ...
        "dense_train_dataset": "train_dataset"
    }
    ...
  ```


## Usage

### Usage: Directory setting
Server의 디렉토리 구조에서 baseline code가 input과 같은 수준에 위치하면 됩니다.

```
root/  
├── input/
└── odqa_baseline_code/  
```

`input` 디렉토리에, 아래와 같이 `input/checkpoint`, `input/data`, `input/embed` 디렉토리를 생성해주세요.

```
input/
├── checkpoint/ - checkpoints&predictions (strategy_alias_seed)
├── data/ - competition data
├── embed/ - embedding caches of wikidocs.json
└── (optional) keys/ - secret keys or tokens
```

**Slack 알람 봇을 활용하시려면** `input/keys`에 `secrets.json`을 넣어주시고, `--report` argument를 `True`로 설정해주세요.    
`secrets.json`은 아래와 같은 형식으로 작성해주세요.  
  
```
{
    "SLACK": {
        "CHANNEL_ID": "[Slack 채널 ID]",
        "TOKEN": "[Slack 채널 토큰]",
        "USER_NAME": "[닉네임]",
        "COLOR": "[hex color]" i.e., "#FFFFFF", 
        "EMOJI": "[Emoji code]" i.e., ":dog:"
    }
}
```
  
**Slack 알람 봇을 사용하지 않으실 경우 해당 디렉토리 및 파일은 만들지 않으셔도 됩니다.**  
  
### Usage: Train   
  
#### Reader
아래 스크립트를 실행하여 Reader 모델의 학습 및 평가를 진행합니다.

```
./scripts/run_mrc.sh
```

- 전략 config의 Retriever 모델은 사용하지 않습니다. MRC 모델 학습시에는 정답 문서를 reader 모델에 바로 제공합니다.
- 실행하면 아래와 같이 checkpoint와 validation set에 대한 결과파일이 생성됩니다.  
- config 파일에서 `train.pororo_prediction` argument를 `True`로 주면 `pororo` 라이브러리로 예측값이 보완된 `pororo_predictions_test.json`이 함께 생성됩니다.

```
input/  
└── checkpoint/  
    ├── ST02_temp_95/
    │   ├── checkpoint-500/
    │   └── ...
    ├── nbest_predictions_valid.json
    ├── predictions_valid.json
    ├── (optional) pororo_predictions_test.json
    └── valid_results.json
```

#### Retriever
아래 스크립트를 실행하여 Retriever 모델의 학습 및 평가를 진행합니다.

```
./scripts/run_retrieval.sh
```

- 전략 config의 Reader 모델은 사용하지 않습니다. 문서 검색을 위한 retriever 모델만을 학습 및 평가합니다.
- 전략 config에서 `retriever.retrain` argument를 `True`로 주면 retriever의 embedding을 재학습시킬 수 있습니다.
- 실행하면 아래와 같이 wandb.ai에서 결과값을 확인 할 수 있습니다.

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

### Usage: Validation
아래 스크립트를 실행하여 종합적인 ODQA 프로세스의 성능 평가를 진행합니다.

```
./scripts/run.sh
```

- Reader와 Retriever를 동시에 활용하여 ODQA 성능을 종합적으로 검증합니다.
- 기학습된 파일들을 불러오기 때문에, train은 진행하지 않고 validation score만 측정합니다.
- 검증 단계이므로 strategies로써 한 개의 전략만 집어넣는 것을 추천합니다.  

- 아래와 같이 전략명에 대한 디렉토리와 파일이 생성됩니다.
- config 파일에서 `train.pororo_prediction` argument를 `True`로 주면 `pororo` 라이브러리로 예측값이 보완된 `pororo_predictions_test.json`이 함께 생성됩니다.
 
```
input/  
└── checkpoint/  
    ├── ST02_temp_95/
    │   ├── checkpoint-500/
    │   └── ...
    ├── nbest_predictions_valid.json
    ├── predictions_valid.json
    ├── (optional) pororo_predictions_test.json
    └── valid_results.json
```

### Usage: Predict
아래 스크립트를 실행하여 학습된 모델을 불러와 예측(predict)을 진행합니다.   

```
./scripts/predict.sh
```    

- Reader와 Retriever를 동시에 활용하여 prediction을 진행합니다.
- 예측에 활용할 전략 한개만 활용할 것을 추천합니다.  
- **예측을 위해서는 예측 대상인 질문 dataset과 retrieval의 대상인 wikipedia document dataset이 필요합니다.** 자세한 내용은 [데이터셋 설정](#dataset-setting)을 참조해주세요.
  
- 예측 결과로 아래와 같이 전략명에 대한 디렉토리와 파일이 생성됩니다.
- config 파일에서 `train.pororo_prediction` argument를 `True`로 주면 `pororo` 라이브러리로 예측값이 보완된 `pororo_predictions_test.json`이 함께 생성됩니다.

```
input/  
└── checkpoint/  
    └── ST01/
        ├── nbest_predictions_test.json
        ├── predictions_test.json
        ├── valid_results.json
        └── (optional) pororo_predictions_test.json
```
  
### Usage: Make additional dataset
부가적인 데이터셋을 생성합니다.    
일부 데이터셋은 생성 이전에 앞서 언급한 **커스텀 데이터셋**을 필요로 합니다. 

```bash
python -m make_dataset.qd_pair_bm25
python -m make_dataset.cheat_dataset
python -m make_dataset.aggregate_wiki
python -m make_dataset.triplet_dataset
python -m make_dataset.kor_sample_dataset
python -m make_dataset.negative_ctxs_dataset
```

## Test Driven Development
- [tester.py](./utils/tester.py): 구현된 기능이 정상 작동되는지 테스트합니다.    

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

## Contributors
[구건모(ggm1207)](https://github.com/ggm1207) | [김성익(SeongIkKim)](https://github.com/SeongIkKim) | [김종헌(olenmg)](https://github.com/olenmg) | [신지영(ebbunnim)](https://github.com/ebbunnim) | [이수연(sooyounlee)](https://github.com/sooyounlee)

## Reference
### Papers
- [Using the Hammer Only on Nails: A Hybrid Method for Evidence Retrieval for Question Answering](https://arxiv.org/abs/2009.10791)
- [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906)
- [Pyserini: An Easy-to-Use Python Toolkit to Support Replicable IR Research with Sparse and Dense Representations](https://arxiv.org/abs/2102.10073)
- [Improvements to BM25 and Language Models Examined](http://www.cs.otago.ac.nz/homepages/andrew/papers/2014-2.pdf)
- [SpanBERT: Improving Pre-training by Representing and Predicting Spans](https://arxiv.org/abs/1907.10529)
- [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT (SIGIR'20).](https://arxiv.org/abs/2004.12832)
- [Reading Wikipedia to Answer Open-Domain Questions](https://arxiv.org/abs/1704.00051)

### Dataset
- [KorQuAD 1.0](https://korquad.github.io/KorQuad%201.0/)
- [(ko)Wikipedia](https://bit.ly/3uoGPCg)

## License
`VumBleBot/odqa_baseline_code`는 **Apache License 2.0**을 따릅니다.  
