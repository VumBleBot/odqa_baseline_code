import os.path as p

from reader import DprReader
from retrieval.sparse import SparseRetrieval
from tokenization_kobert import KoBertTokenizer
from datasets import load_from_disk, load_dataset, load_metric
from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer


metric = load_metric("squad")

READER = {"DPR": DprReader}


def get_retriever(args):
    """
    Get appropriate retriever.

    AVAILABLE OPTIONS(2021.05.02)
    - Term-based
        - TF-IDF : use konlpy-Mecab for word tokenization.
        - TODO : BM-25
    - Vector Embedding
        - Sparse
        - TODO : Dense
    Need more retriever and retriever options.

    :param args
        - model.retriever_name : [tfidf]
    :return: Retriever which contains embedded vector(+indexer if faiss is built).
    """
    if args.model.retriever_name == "tfidf":
        from konlpy.tag import Mecab

        mecab = Mecab()
        retriever = SparseRetrieval(args, tokenize_fn=mecab.morphs)
        retriever.get_sparse_embedding()

    return retriever


def get_reader(args, datasets):
    """
    Get pretrained MRC-Reader model and tokenizer.
    If model setting is KoBERT, then load tokenizer from public KoBERT Tokenizer.
    Else, transformers library autosets appropriate tokenizer from model(when tokenizer is not specified).

    :param args:
        - model.model_name_or_path(required) : model repository name that registered in huggingface library.
            (ex - 'monologg/koelectra-small-v3-discriminator')
        - model.model_path : saved checkpoint in server disk.
            (ex - '/input/checkpoint/ST01_0_temp/checkpoint-500')
        - model.config_name
        - model_tokenizer_name
    :return: pretrained model and tokenizer.
    """
    config = AutoConfig.from_pretrained(
        args.model.config_name if args.model.config_name else args.model.model_name_or_path
    )

    if args.model.model_name_or_path in ["monologg/kobert", "monologg/distilkobert"]:
        # if args.model_path != "" then load from args.model_path
        tokenizer = KoBertTokenizer.from_pretrained(args.model_path or args.model.model_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model.tokenizer_name if args.model.tokenizer_name else args.model.model_name_or_path, use_fast=True
        )

    model = AutoModelForQuestionAnswering.from_pretrained(
        args.model.model_name_or_path, from_tf=bool(".ckpt" in args.model.model_name_or_path), config=config
    )

    reader = READER[args.model.reader_name](args, model, tokenizer, datasets)

    return reader


def get_dataset(args, is_train=True):
    """
    Load dataset from dataset path in disk.

    :param args
        - data.dataset_name : [train_dataset, test_dataset, squad_kor_v1]
        - debug : True expressions. If this setting is true, epoch and dataset will be restricted for quick testing.
    :param is_train: True for training, False for validation.
    :return: Loaded dataset.
    """
    datasets = None

    if args.data.dataset_name == "train_dataset":
        if is_train:
            datasets = load_from_disk(p.join(args.path.train_data_dir, args.data.dataset_name))
        else:
            datasets = load_from_disk(p.join(args.path.train_data_dir, "test_dataset"))
    elif args.data.dataset_name == "squad_kor_v1":
        datasets = load_dataset(args.data.dataset_name)
    # Add more dataset option here.

    if datasets is None:
        raise KeyError(f"{args.data.dataset_name}데이터는 존재하지 않습니다.")

    if args.debug:
        args.train.num_train_epochs = 1.0
        datasets["train"] = datasets["train"].select(range(100))

    return datasets
