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


class QuestionAnsweringTrainer(Trainer):
    def __init__(self, *args, custom_args=None, eval_examples=None, post_process_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function

    # def evaluate(self, eval_dataset=None, eval_examples=None, ignore_keys=None):
    #     eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
    #     eval_dataloader = self.get_eval_dataloader(eval_dataset)
    #     eval_examples = self.eval_examples if eval_examples is None else eval_examples
    #
    #     # Temporarily disable metric computation, we will do it in the loop here.
    #     compute_metrics = self.compute_metrics
    #     self.compute_metrics = None
    #     try:
    #         output = self.prediction_loop(
    #             eval_dataloader,
    #             description="Evaluation",
    #             # No point gathering the predictions if there are no metrics, otherwise we defer to
    #             # self.args.prediction_loss_only
    #             prediction_loss_only=True if compute_metrics is None else None,
    #             ignore_keys=ignore_keys,
    #         )
    #     finally:
    #         self.compute_metrics = compute_metrics
    #
    #     ### 아웃풋만 남음
    #
    #     # We might have removed columns from the dataset so we put them back.
    #     if isinstance(eval_dataset, datasets.Dataset):
    #         eval_dataset.set_format(type=eval_dataset.format["type"], columns=list(eval_dataset.features.keys()))
    #
    #     if self.post_process_function is not None and self.compute_metrics is not None:
    #         eval_preds, use_pororo = self.post_process_function(eval_examples, eval_dataset, output.predictions, self.args)
    #         metrics = self.compute_metrics(eval_preds, use_pororo=use_pororo)
    #
    #         # pororo를 사용할 경우 pororo_voted_prediction만, pororo를 사용하지 않을 경우 original prediction만을 이용하여 측정
    #         self.log(metrics[-1])
    #     else:
    #         metrics = {}
    #
    #     self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
    #     return metrics
    #
    # def predict(self, test_dataset, test_examples, ignore_keys=None):
    #     test_dataloader = self.get_test_dataloader(test_dataset)
    #
    #     # Temporarily disable metric computation, we will do it in the loop here.
    #     compute_metrics = self.compute_metrics # tuple object
    #     self.compute_metrics = None
    #     try:
    #         output = self.prediction_loop(
    #             test_dataloader,
    #             description="Evaluation",
    #             # No point gathering the predictions if there are no metrics, otherwise we defer to
    #             # self.args.prediction_loss_only
    #             prediction_loss_only=True if compute_metrics is None else None,
    #             ignore_keys=ignore_keys,
    #         )
    #     finally:
    #         self.compute_metrics = compute_metrics
    #
    #     if self.post_process_function is None or self.compute_metrics is None:
    #         return output
    #
    #     # We might have removed columns from the dataset so we put them back.
    #     if isinstance(test_dataset, datasets.Dataset):
    #         test_dataset.set_format(type=test_dataset.format["type"], columns=list(test_dataset.features.keys()))
    #
    #     predictions = self.post_process_function(test_examples, test_dataset, output.predictions, self.args)
    #     return predictions

    # 고친 버전

    # kfold마다 n번 돌려서 output을 얻어낼 함수
    def get_prediction_output(self, mode, dataset=None, examples=None, ignore_keys=None):
        if mode == 'eval':
            eval_dataset = self.eval_dataset if dataset is None else dataset
            dataloader = self.get_eval_dataloader(eval_dataset)
            examples = self.eval_examples if examples is None else examples
        elif mode == 'predict':
            test_dataset = dataset
            dataloader = self.get_test_dataloader(test_dataset)
        else:
            print("mode를 'eval' 혹은 'predict'로 명시해주세요.")
            raise KeyError()

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.prediction_loop(
                dataloader,
                description="Evaluation" if mode == 'eval' else "Prediction",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        return output

    # 기존 evaluate 역할
    # + kfold_output으로 모든 output을 합친 값이 들어온다면 get_prediction_output하지 않고 바로 처리.
    def evaluate(self, eval_dataset=None, eval_examples=None, kfold_output=None):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        if kfold_output == None:
            # kfold를 하지 않을때도 원래 flow대로 처리
            output = self.get_prediction_output(mode='eval', dataset=eval_dataset, examples=eval_examples)
        else:
            output = kfold_output  # mean값으로 처리한 output값이 들어와야함

        if isinstance(eval_dataset, datasets.Dataset):
            eval_dataset.set_format(type=eval_dataset.format["type"], columns=list(eval_dataset.features.keys()))

        if self.post_process_function is not None and self.compute_metrics is not None:
            eval_preds, use_pororo = self.post_process_function(eval_examples, eval_dataset, output.predictions, self.args)

            metrics = self.compute_metrics(eval_preds, use_pororo=use_pororo)
            if isinstance(metrics, tuple):
                metrics = metrics[-1]

            self.log(metrics)
        else:
            metrics = {}

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics

    # 기존 predict 역할
    def predict(self, test_dataset, test_examples):
        test_dataloader = self.get_test_dataloader(test_dataset)

        output = self.get_prediction_output(mode='predict', datasets=test_dataset)

        if self.post_process_function is None or self.compute_metrics is None:
            return output

        # We might have removed columns from the dataset so we put them back.
        if isinstance(test_dataset, datasets.Dataset):
            test_dataset.set_format(type=test_dataset.format["type"], columns=list(test_dataset.features.keys()))

        predictions = self.post_process_function(test_examples, test_dataset, output.predictions, self.args)
        return predictions

    # kfold_output으로 모든 output 합친 값이 들어온다면 get_prediction_output하지 않고 바로 처리
    def kfold_predict(self, test_dataset, test_examples, kfold_output):

        if self.post_process_function is None or self.compute_metrics is None:
            return kfold_output

        # We might have removed columns from the dataset so we put them back.
        if isinstance(test_dataset, datasets.Dataset):
            test_dataset.set_format(type=test_dataset.format["type"], columns=list(test_dataset.features.keys()))

        predictions = self.post_process_function(test_examples, test_dataset, kfold_output, self.args)
        return predictions
