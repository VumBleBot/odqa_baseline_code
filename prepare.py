import os.path as p

from tokenization_kobert import KoBertTokenizer
from utils_qa import postprocess_qa_predictions
from datasets import load_from_disk, load_dataset, load_metric
from transformers import EvalPrediction
from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer
from retriever.sparse import SparseRetrieval


metric = load_metric("squad")


def compute_metrics(p):
    return metric.compute(predictions=p.predictions, references=p.label_ids)


def get_retriever(args):
    if args.model.retriever_name == "tfidf":
        from konlpy.tag import Mecab

        mecab = Mecab()
        retriever = SparseRetrieval(args, tokenize_fn=mecab.morphs)
    return retriever


def get_reader_model(args):
    config = AutoConfig.from_pretrained(
        args.model.config_name if args.model.config_name else args.model.model_name_or_path
    )

    if args.model.model_name_or_path in ["monologg/kobert", "monologg/distilkobert"]:
        tokenizer = KoBertTokenizer.from_pretrained(args.model.model_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model.tokenizer_name if args.model.tokenizer_name else args.model.model_name_or_path, use_fast=True
        )

    model = AutoModelForQuestionAnswering.from_pretrained(
        args.model.model_name_or_path, from_tf=bool(".ckpt" in args.model.model_name_or_path), config=config
    )

    return model, tokenizer


def prepare_dataset(args, is_train=True):
    datasets = None

    if args.data.dataset_name == "train_dataset":
        if is_train:
            datasets = load_from_disk(p.join(args.path.train_data_dir, args.data.dataset_name))
        else:
            datasets = load_from_disk(p.join(args.path.train_data_dir, "test_dataset"))
    elif args.data.dataset_name == "squad_kor_v1":
        datasets = load_dataset(args.data.dataset_name)

    if datasets is None:
        raise KeyError(f"{args.data.dataset_name}데이터는 존재하지 않습니다.")

    if args.debug:
        args.train.num_train_epochs = 1.0
        datasets["train"] = datasets["train"].select(range(100))

    return datasets


def preprocess_dataset(args, datasets, tokenizer, is_train=True):
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
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
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

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        return tokenized_examples

    def post_processing_function(examples, features, predictions, training_args):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            max_answer_length=args.data.max_answer_length,
            output_dir=training_args.output_dir,
            prefix="test" if args.train.do_predict else "valid",
        )
        # Format the result to the format the metric expects.
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
