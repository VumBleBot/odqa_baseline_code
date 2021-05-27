# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A subclass of `Trainer` specific to Question-Answering tasks
"""

import datasets
from transformers import Trainer

from utils.utils_qa import get_logits_with_offset


class QuestionAnsweringTrainer(Trainer):
    def __init__(self, *args, custom_args=None, eval_examples=None, post_process_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_args = custom_args
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function

    def evaluate(self, eval_dataset=None, eval_examples=None, ignore_keys=None):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.prediction_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        # We might have removed columns from the dataset so we put them back.
        if isinstance(eval_dataset, datasets.Dataset):
            eval_dataset.set_format(type=eval_dataset.format["type"], columns=list(eval_dataset.features.keys()))

        metric_results = {}

        if self.post_process_function is not None and self.compute_metrics is not None:
            valid_results = self.post_process_function(eval_examples, eval_dataset, output.predictions, self.args)

            for pred_type, eval_preds in valid_results.items():
                metrics = self.compute_metrics(eval_preds)
                metric_results[pred_type] = metrics

                # Logging용 metrics
                metrics = {f"{pred_type}_{k}": v for k, v in metrics.items()}
                self.log(metrics)  # logs: Dict[str, float]
        else:
            metrics = {}

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metric_results

    def get_logits_with_keys(self, test_dataset, test_examples, keys=None, ignore_keys=None):
        for key in keys:
            assert key in test_examples.features.keys(), f"{key}는 {test_examples.features}안에 없습니다!"

        test_dataloader = self.get_test_dataloader(test_dataset)

        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.prediction_loop(
                test_dataloader,
                description="Evaluation",
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if isinstance(test_dataset, datasets.Dataset):
            test_dataset.set_format(type=test_dataset.format["type"], columns=list(test_dataset.features.keys()))

        logits = get_logits_with_offset(
            test_examples,
            test_dataset,
            output.predictions,
            topk=self.custom_args.retriever.topk,  # custom_args: args, returned to tools.get_args()
            max_answer_length=self.custom_args.data.max_answer_length,
        )

        return logits, (list(test_examples[k]) for k in keys)

    def predict(self, test_dataset, test_examples, ignore_keys=None):
        test_dataloader = self.get_test_dataloader(test_dataset)

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics  # tuple object
        self.compute_metrics = None
        try:
            output = self.prediction_loop(
                test_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is None or self.compute_metrics is None:
            return output

        # We might have removed columns from the dataset so we put them back.
        if isinstance(test_dataset, datasets.Dataset):
            test_dataset.set_format(type=test_dataset.format["type"], columns=list(test_dataset.features.keys()))

        predictions = self.post_process_function(test_examples, test_dataset, output.predictions, self.args)
        return predictions
