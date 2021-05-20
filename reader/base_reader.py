from transformers import TrainerCallback, EvalPrediction, DataCollatorWithPadding

from datasets import load_metric
from utils_qa import postprocess_qa_predictions
from trainer_qa import QuestionAnsweringTrainer


class BaseReader:
    def __init__(self, args, model, tokenizer, eval_answers):
        self.args = args
        self.metric = load_metric("squad")
        self.model, self.tokenizer = model, tokenizer

        self.data_collator = DataCollatorWithPadding(
            self.tokenizer, pad_to_multiple_of=8 if self.args.train.fp16 else None
        )

        self.train_dataset = None # 필요한거: train pp된거, eval 원본, eval retrieve 된거, eval retrieve 되고 pp된거
        self.eval_dataset = None
        self.eval_examples = None
        self.eval_answers = eval_answers

    def set_dataset(self, train_dataset=None, eval_dataset=None):
        if train_dataset:
            self.train_dataset = self.preprocess_dataset(train_dataset, is_train=True)
        if eval_dataset:
            self.eval_dataset = self.preprocess_dataset(eval_dataset, is_train=False)
            self.eval_examples = eval_dataset

    def preprocess_dataset(self, dataset, is_train=True):
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
        column_names = dataset.column_names

        self.question_column_name = "question" if "question" in column_names else column_names[0]
        self.context_column_name = "context" if "context" in column_names else column_names[1]
        self.answer_column_name = "answers" if "answers" in column_names else column_names[2]

        self.pad_on_right = self.tokenizer.padding_side == "right"
        self.max_seq_length = min(self.args.data.max_seq_length, self.tokenizer.model_max_length)

        prepare_func = self._prepare_train_features if is_train else self._prepare_validation_features

        dataset = dataset.map(
            prepare_func,
            batched=True,
            batch_size=self.args.data.batch_size,
            num_proc=self.args.data.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not self.args.data.overwrite_cache,
            cache_file_name=self.args.data.cache_file_name,
        )

        return dataset

    def _prepare_train_features(self, examples):
        tokenized_examples = self.tokenizer(
            examples[self.question_column_name if self.pad_on_right else self.context_column_name],
            examples[self.context_column_name if self.pad_on_right else self.question_column_name],
            truncation="only_second" if self.pad_on_right else "only_first",
            max_length=self.max_seq_length,
            stride=self.args.data.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if self.args.data.pad_to_max_length else False,
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_examples.pop("offset_mapping")

        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            sequence_ids = tokenized_examples.sequence_ids(i)

            sample_index = sample_mapping[i]
            answers = examples[self.answer_column_name][sample_index]

            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if self.pad_on_right else 0):
                    token_start_index += 1

                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if self.pad_on_right else 0):
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

    def _prepare_validation_features(self, examples):
        tokenized_examples = self.tokenizer(
            examples[self.question_column_name if self.pad_on_right else self.context_column_name],
            examples[self.context_column_name if self.pad_on_right else self.question_column_name],
            truncation="only_second" if self.pad_on_right else "only_first",
            max_length=self.max_seq_length,
            stride=self.args.data.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if self.args.data.pad_to_max_length else False,
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if self.pad_on_right else 0

            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    def _post_processing_function(self, examples, features, predictions, training_args):
        predictions, pororo_predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            topk=self.args.retriever.topk,
            max_answer_length=self.args.data.max_answer_length,
            output_dir=training_args.output_dir,
            prefix="test" if self.args.train.do_predict else "valid",
        )

        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
        if training_args.pororo_prediction:
            formatted_pororo_predictions = [{"id": k, "prediction_text": v} for k, v in pororo_predictions.items()]

            if training_args.do_predict:
                return formatted_predictions

            elif training_args.do_eval:
                # query, 정답
                references = [
                    {"id": ex["id"], "answers": ex[self.answer_column_name]} for ex in self.eval_answers
                ]
                return (
                    EvalPrediction(predictions=formatted_predictions, label_ids=references),
                    EvalPrediction(predictions=formatted_pororo_predictions, label_ids=references)
                ), training_args.pororo_prediction

        if training_args.do_predict:
            return formatted_predictions

        elif training_args.do_eval:
            # query, 정답
            references = [
                {"id": ex["id"], "answers": ex[self.answer_column_name]} for ex in self.eval_answers
            ]
            return EvalPrediction(predictions=formatted_predictions, label_ids=references), training_args.pororo_prediction

    def _compute_metrics(self, p, use_pororo=False):
        if use_pororo:
            p, pororo_p = p
            return (
                self.metric.compute(predictions=p.predictions, references=p.label_ids),
                self.metric.compute(predictions=pororo_p.predictions, references=pororo_p.label_ids)
            )

        return (self.metric.compute(predictions=p.predictions, references=p.label_ids),)

    def get_trainer(self):
        raise NotImplementedError


class EvalCallback(TrainerCallback):
    def on_step_end(self, args, state, control,**kwargs):
        if args.do_eval_during_training and state.global_step % args.eval_step == 0:
            control.should_evaluate = True
            # if args.load_best_model_at_end:
            #     control.should_save = True
        
        return control


class DprReader(BaseReader):
    def __init__(self, args, model, tokenizer, eval_answers):
        super().__init__(args, model, tokenizer, eval_answers)

    def get_trainer(self):
        trainer = QuestionAnsweringTrainer(
            model=self.model,
            args=self.args.train,  # training_args
            custom_args=self.args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            eval_examples=self.eval_examples,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            post_process_function=self._post_processing_function,
            compute_metrics=self._compute_metrics,
            #callbacks=[EvalCallback]
        )

        return trainer
