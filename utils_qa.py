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
Post-processing utilities for question answering.
"""

import os
import json
import collections
from typing import Optional, Tuple

import numpy as np
from tqdm.auto import tqdm
from konlpy.tag import Mecab

mecab = Mecab()


def tokenize(text):
    # return text.split(" ")
    return mecab.morphs(text)

def looping_through_all_features(
        all_start_logits, all_end_logits, n_best_size, features, max_answer_length, feature_indices
):
    min_null_prediction = None
    prelim_predictions = []
    for feature_index in feature_indices:
        start_logits = all_start_logits[feature_index]
        end_logits = all_end_logits[feature_index]
        offset_mapping = features[feature_index]["offset_mapping"]
        token_is_max_context = features[feature_index].get("token_is_max_context", None)
        feature_null_score = start_logits[0] + end_logits[0]
        if min_null_prediction is None or min_null_prediction["score"] > feature_null_score:
            min_null_prediction = {
                "offsets": (0, 0),
                "score": feature_null_score,
                "start_logit": start_logits[0],
                "end_logit": end_logits[0],
            }
        start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
        end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
        for start_index in start_indexes:
            for end_index in end_indexes:
                if (
                    start_index >= len(offset_mapping)
                    or end_index >= len(offset_mapping)
                    or offset_mapping[start_index] is None
                    or offset_mapping[end_index] is None
                ):
                    continue
                if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                    continue
                if token_is_max_context is not None and not token_is_max_context.get(str(start_index), False):
                    continue
                prelim_predictions.append(
                    {
                        "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                        "score": start_logits[start_index] + end_logits[end_index],
                        "start_logit": start_logits[start_index],
                        "end_logit": end_logits[end_index],
                    }
                )
    return prelim_predictions



def postprocess_qa_predictions(
        examples,
        features,
        predictions: Tuple[np.ndarray, np.ndarray],
        topk: int = 1,
        version_2_with_negative: bool = False,
        n_best_size: int = 20,
        max_answer_length: int = 30,
        null_score_diff_threshold: float = 0.0,
        output_dir: Optional[str] = None,
        prefix: Optional[str] = None,
        is_world_process_zero: bool = True,
):
    """
    Post-processes the predictions of a question-answering model to convert them to answers that are substrings of the
    original contexts. This is the base postprocessing functions for models that only return start and end logits.
    Args:
        examples: The non-preprocessed dataset (see the main script for more information).
        features: The processed dataset (see the main script for more information).
        predictions (:obj:`Tuple[np.ndarray, np.ndarray]`):
            The predictions of the model: two arrays containing the start logits and the end logits respectively. Its
            first dimension must match the number of elements of :obj:`features`.
        version_2_with_negative (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the underlying dataset contains examples with no answers.
        n_best_size (:obj:`int`, `optional`, defaults to 20):
            The total number of n-best predictions to generate when looking for an answer.
        max_answer_length (:obj:`int`, `optional`, defaults to 30):
            The maximum length of an answer that can be generated. This is needed because the start and end predictions
            are not conditioned on one another.
        null_score_diff_threshold (:obj:`float`, `optional`, defaults to 0):
            The threshold used to select the null answer: if the best answer has a score that is less than the score of
            the null answer minus this threshold, the null answer is selected for this example (note that the score of
            the null answer for an example giving several features is the minimum of the scores for the null answer on
            each feature: all features must be aligned on the fact they `want` to predict a null answer).
            Only useful when :obj:`version_2_with_negative` is :obj:`True`.
        output_dir (:obj:`str`, `optional`):
            If provided, the dictionaries of predictions, n_best predictions (with their scores and logits) and, if
            :obj:`version_2_with_negative=True`, the dictionary of the scores differences between best and null
            answers, are saved in `output_dir`.
        prefix (:obj:`str`, `optional`):
            If provided, the dictionaries mentioned above are saved with `prefix` added to their names.
        is_world_process_zero (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether this process is the main process or not (used to determine if logging/saves should be done).
    """
    assert len(predictions) == 2, "`predictions` should be a tuple with two elements (start_logits, end_logits)."
    all_start_logits, all_end_logits = predictions

    assert len(predictions[0]) == len(
        features), f"Got {len(predictions[0])} predictions and {len(features)} features."

    # Build a map example to its corresponding features.
    # example_id_to_index = {'mrc-0-00XXXX_0' : 0, 'mrc-0-00XXXX_1' : 1, ....} --> 720개 example마다 각각 다른 example_id를 준다.
    example_id_to_index = {'_'.join([k, str(i % topk)]): i for i, k in enumerate(examples["id"])}

    features_per_example = collections.defaultdict(list)
    # ex) features_per_example[0] ==> [0], features_per_example[0] ==> [1,2,3] ....
    prev_doc_offset = (-1, -1)[0]
    doc_id_postfix = 0
    for i, feature in enumerate(features):
        # token_type_ids로 query sequence 구한 뒤, 거기서부터 뒤로 일정부분 슬라이싱해 docs 구분하기
        tti = feature['token_type_ids']
        query_sequence_length = len(tti) - sum(tti)  # query sequence length
        doc_offset = feature['offset_mapping'][query_sequence_length][0]  # 해당 context sequence의 첫번째 offset

        # offset이 떨어지거나 같으면(0) --> topk묶음이 끝나면
        if doc_offset <= prev_doc_offset:
            # doc_id_postfix가 0~topk-1까지 가도록 조정.
            if (doc_id_postfix + 1) % topk == 0:
                doc_id_postfix = 0
            else:
                doc_id_postfix += 1
            # example_id_to_index의 키값으로 사용할 문자열 조합
            # ex) mrc-00-00XXXX_0, mrc-00-00XXXX_2

        # 해당 feature를 example index dict에 등록
        example_index_key = '_'.join([feature['example_id'], str(doc_id_postfix)])
        features_per_example[example_id_to_index[example_index_key]].append(i)

        prev_doc_offset = doc_offset

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()

    topk_merged_predictions = []

    # 시작
    # example은 len(example_to_index) * topk 번을 돈다.
    # 따라서 각 topk 묶음의 첫번째 인덱스인 bundle_start_index를 활용.
    for bundle_start_index in tqdm(range(0, len(examples), topk)):
        for example_index in range(bundle_start_index, bundle_start_index + topk):
            example = examples[example_index]

            feature_indices = features_per_example[example_index]

            # print(f"example {example_index} | feature_indices {feature_indices}")
            # print(f"example {example['question']} | feature_indices {feature_indices}")


            # 하나의 example에 딸린 context들을 전부 돌면서 prediction 수집
            prelim_predictions = looping_through_all_features(
                all_start_logits, all_end_logits, n_best_size, features, max_answer_length, feature_indices
            )  # [offset, start logit, end logit, score]


            # TODO: document마다 score 정규화

            # 한 example에 있는 모든 predictions.
            predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

            # 01 predictions 정답 텍스트 매핑
            context = example["context"]
            for pred in predictions:
                offsets = pred.pop("offsets")
                pred["text"] = context[offsets[0]: offsets[1]]
            # 02 정답이 없다면 Fake 정답 생성
            if len(predictions) == 0 or (len(predictions) == 1 and predictions[0]["text"] == ""):
                predictions.insert(0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0})
            # 03 확률값 계산(softmax)
            scores = np.array([pred["score"] for pred in predictions])
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / exp_scores.sum()
            for prob, pred in zip(probs, predictions):
                pred["probability"] = prob
                # prediction 결과분석용
                pred["context_id"] = example['context_id']
                pred["context"] = example['context']


            topk_merged_predictions.extend(predictions)

            # print(f"len(predictions): {len(predictions)} | len(topk_merged_predictions) : {len(topk_merged_predictions)} | len(prelimed) : {len(prelim_predictions)}")


        # 아래는 topk개로 묶어서 수행하는 logic

        # (빈 답 거르고 softmax한 probability로 계산?)
        # topk_merged_predictions = [pred for pred in topk_merged_predictions if pred["probability"] != 1.0]

        # 1. score로 계산
        predictions = sorted(topk_merged_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

        # 정답 로깅
        all_predictions[example["id"]] = predictions[0]["text"]
        # 정답 포함 가능성 있었던 답을 로깅
        all_nbest_json[example["id"]] = [
            {k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v) for k, v in
             pred.items()}
            for pred in predictions
        ]

        # print(f"{bundle_start_index+topk+1}, {topk} --- FLUSH")
        topk_merged_predictions = []  # k개 context를 묶었다가 flush

    # 그럼 이제 저장
    if output_dir is not None:
        assert os.path.isdir(output_dir), f"{output_dir} is not a directory."
        prediction_file = os.path.join(
            output_dir, "predictions.json" if prefix is None else f"predictions_{prefix}.json"
        )
        nbest_file = os.path.join(
            output_dir, "nbest_predictions.json" if prefix is None else f"nbest_predictions_{prefix}.json"
        )
        with open(prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=4, ensure_ascii=False) + "\n")
        with open(nbest_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4, ensure_ascii=False) + "\n")
    return all_predictions
