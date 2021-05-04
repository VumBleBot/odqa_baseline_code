import os.path as p

from tokenization_kobert import KoBertTokenizer
from utils_qa import postprocess_qa_predictions
from datasets import load_from_disk, load_dataset, load_metric
from transformers import EvalPrediction
from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer
# from retrieval.sparse import SparseRetrieval
from retrieval.sparse.sparse_base import TfidfRetrieval


metric = load_metric("squad")


def compute_metrics(p):
    return metric.compute(predictions=p.predictions, references=p.label_ids)


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
        # retriever = SparseRetrieval(args, tokenize_fn=mecab.morphs)
        # retriever.get_sparse_embedding()
        retriever = TfidfRetrieval(args, tokenize_fn=mecab.morphs)
    retriever.get_embedding()
    return retriever


def get_reader_model(args):
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

    return model, tokenizer


def prepare_dataset(args, is_train=True):
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


def preprocess_dataset(args, datasets, tokenizer, is_train=True):
    """
    Setup dataset for training/validation/inference.
    Inner functions
        - prepare_train_features : for train dataset. tokenizing, offset mapping + answer position that model predicts.
        - prepare_validation_feature : for validation dataset.  tokenizing, offset mapping.

    :param args
        - data.max_seq_length
        - data.doc_stride
        - data.pad_to_max_length
        - max_answer_length
    :param datasets
    :param tokenizer
    :param is_train : [True, False]
        use prepare_train_features function if True. else, use prepare_validation_features function.
    :return: (preprocessed dataset, postprocessing function for prediction)
    """
    data_type = "train" if is_train else "validation"

    column_names = datasets[data_type].column_names

    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    pad_on_right = tokenizer.padding_side == "right"
    max_seq_length = min(args.data.max_seq_length, tokenizer.model_max_length)

    def prepare_train_features(examples):
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=args.data.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if args.data.pad_to_max_length else False,
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_examples.pop("offset_mapping")

        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            sequence_ids = tokenized_examples.sequence_ids(i)

            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]

            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    def prepare_validation_features(examples):
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=args.data.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if args.data.pad_to_max_length else False,
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        return tokenized_examples

    def post_processing_function(examples, features, predictions, training_args):
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            max_answer_length=args.data.max_answer_length,
            output_dir=training_args.output_dir,
            prefix="test" if args.train.do_predict else "valid",
        )

        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
        if training_args.do_predict:
            return formatted_predictions

        elif training_args.do_eval:
            references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in datasets["validation"]]
            return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    dataset = datasets[data_type]
    prepare_func = prepare_train_features if is_train else prepare_validation_features

    dataset = dataset.map(
        prepare_func,
        batched=True,
        batch_size=args.data.batch_size,
        num_proc=args.data.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not args.data.overwrite_cache,
    )

    return dataset, post_processing_function
