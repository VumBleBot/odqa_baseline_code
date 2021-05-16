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
import copy
import os
import json
import collections
from typing import Optional, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm
from konlpy.tag import Mecab

from reader.pororo_reader import PororoMrcFactory

mecab = Mecab()

# TODO alpha를 config로 빼기
# pororo voting 가중
alpha = 2.0

def check_predictions_and_features(predictions, features):
    """
    Check assertions for predictions ans features length.
    If passes, return all start/end logits.

    :param predictions: predictions
    :param features: tokenized and divided contexts(The processed dataset, cut by max_sequence_length)
    :return: all start/end logits if pass assertions.
    """
    assert len(predictions) == 2, "`predictions` should be a tuple with two elements (start_logits, end_logits)."
    all_start_logits, all_end_logits = predictions

    assert len(predictions[0]) == len(
        features), f"Got {len(predictions[0])} predictions and {len(features)} features."

    return all_start_logits, all_end_logits


def build_features_per_example_map(examples, topk, features):
    """
    Build {example : features} dictionary for  predictions from topk features.
    return mapped dictionary.

    Document means original context. Feature means tokenized, divided document.
    :ex
        Document 1 = [ Feature1-1, Feature1-2, Feature1-3, ... ]
        Document 2 = [ Feature2-1, Feature2-2 ]

    :param examples: QA Dataset
    :param topk: variable for binding topk documents in one example.
    :param features: tokenized and divided contexts(The processed dataset, cut by max_sequence_length)
    :return: dict of one example(index) - n features.
        - key : one example index(NOT example_id, such as mrc-00-1234)
        - value : n features(divided contexts) index which are related to the example(key).
                  These contexts contains all topk context features.
    """

    # Build a map example to its corresponding features.
    # example_id_to_index = {'mrc-0-00XXXX_0' : 0, 'mrc-0-00XXXX_1' : 1, ....} --> origin*topk개 example마다 각각 다른 example_id를 준다.
    example_id_to_index = {'_'.join([k, str(i % topk)]): i for i, k in enumerate(examples["id"])}

    features_per_example = collections.defaultdict(list)
    # ex) features_per_example[0] ==> [0], features_per_example[0] ==> [1,2,3] ....
    prev_doc_offset = (-1, -1)[0]
    doc_id_postfix = 0
    for i, feature in enumerate(features):

        # query sequence를 지나 document의 첫번째 offset을 가리키는 doc_pointer
        doc_pointer = 0
        while feature['offset_mapping'][doc_pointer] == None:
            doc_pointer += 1
        doc_offset = feature['offset_mapping'][doc_pointer][0]  # 해당 context sequence의 첫번째 offset

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

    return features_per_example



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


def get_all_prelim_predictions(
        examples, features, features_per_example,
        all_start_logits, all_end_logits,
        max_answer_length,
        topk, n_best_size
):
    """
    Return list of predictions(in nbest_size) per context with descending order.

    :param examples: QA Dataset
    :param features: tokenized and divided contexts(The processed dataset, cut by max_sequence_length)
    :param features_per_example: dict of one example(index) - n features.
    :param all_start_logits: all start logits that reader model predicts
    :param all_end_logits: all end logits that reader model predicts
    :param topk: variable for binding topk documents in one example.
    :param n_best_size: The total number of n-best predictions to generate when looking for an answer.
    :return: all raw predictions sorted by score
        - list of dict {'offsets': (start, end), 'score' : int, 'start_logit' : int, 'end_logit' : int}
    """
    all_prelim_predictions = []

    # example은 len(example_to_index) * topk 번을 돈다.
    # 따라서 각 topk 묶음의 첫번째 인덱스인 bundle_start_index를 활용.
    for bundle_start_index in tqdm(range(0, len(examples), topk)):
        for example_index in range(bundle_start_index, bundle_start_index + topk):
            # example = examples[example_index]

            feature_indices = features_per_example[example_index]

            # print(f"example {example_index} | feature_indices {feature_indices}")
            # print(f"example {example['question']} | feature_indices {feature_indices}")

            # 하나의 example에 딸린 context들을 전부 돌면서 prediction 수집
            prelim_predictions = looping_through_all_features(
                all_start_logits, all_end_logits, n_best_size, features, max_answer_length, feature_indices
            )  # [offset, start logit, end logit, score]

            all_prelim_predictions.append(
                sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size])

    return all_prelim_predictions

def make_predictions(examples, all_prelim_predictions, topk):
    """
    Make formatted predictions (NOT SORTED)

    :param examples: QA Dataset
    :param all_prelim_predictions: all raw predictions sorted by score
    :param topk: variable for binding topk documents in one example.
    :return: all predictions  (NOT SORTED, topk * len(dataset))
        - list of predicts
    """
    all_predictions = []

    for bundle_start_index in tqdm(range(0, len(examples), topk)):
        for example_index in range(bundle_start_index, bundle_start_index + topk):
            example = examples[example_index]
            predictions = all_prelim_predictions[example_index]

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
                # run_mrc의 경우 retrieve가 되지 않으므로 document_id(정답 문서 id)만 존재
                # run의 경우 retrieve 과정에서 predict source document를 context_id로 가공하여 전달
                pred["question"] = example['question']
                pred["context_id"] = example['context_id'] if 'context_id' in example.keys() else example['document_id']
                pred["context"] = example['context']

            all_predictions.append(predictions)

    return all_predictions


def select_top_score_predict(examples, all_predictions, n_best_size, topk):

    # initialize
    final_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()

    for bundle_start_index in tqdm(range(0, len(examples), topk)):
        example = examples[bundle_start_index]
        topk_merged_predictions = []
        for example_index in range(bundle_start_index, bundle_start_index + topk):
            topk_merged_predictions.extend(all_predictions[example_index])

        predictions = sorted(topk_merged_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

        # 정답 로깅
        final_predictions[example["id"]] = predictions[0]["text"]

        # 정답 포함 가능성 있었던 답을 로깅
        all_nbest_json[example["id"]] = [
            {k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v) for k, v in
             pred.items()}
            for pred in predictions
        ]
    return final_predictions, all_nbest_json


def save_predictions_to_json(final_predictions, all_nbest_json, output_dir, prefix):
    assert os.path.isdir(output_dir), f"{output_dir} is not a directory."

    prediction_file = os.path.join(
        output_dir, "predictions.json" if prefix is None else f"predictions_{prefix}.json"
    )
    nbest_file = os.path.join(
        output_dir, "nbest_predictions.json" if prefix is None else f"nbest_predictions_{prefix}.json"
    )
    with open(prediction_file, "w") as writer:
        writer.write(json.dumps(final_predictions, indent=4, ensure_ascii=False) + "\n")
    with open(nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4, ensure_ascii=False) + "\n")


def load_predictions_from_json(data_dir, prefix):
    assert os.path.isdir(data_dir), f"{data_dir} is not a directory."

    prediction_file = os.path.join(
        data_dir, "predictions.json" if prefix is None else f"predictions_{prefix}.json"
    )
    nbest_file = os.path.join(
        data_dir, "nbest_predictions.json" if prefix is None else f"nbest_predictions_{prefix}.json"
    )
    with open(prediction_file, "r") as prediction_file:
        predictions = json.load(prediction_file)
    with open(nbest_file, "r") as nbest_file:
        nbests = json.load(nbest_file)

    return predictions, nbests


def remove_last_postposition(predict):
    """
    예측 마지막에 조사가 있다면 제거
    """
    pos_tagged_predict = mecab.pos(predict)
    # 조사제거
    if len(predict) != 0 and pos_tagged_predict[-1][-1].startswith('J'):
        predict = predict.replace(pos_tagged_predict[-1][0], "")
    return predict


def pororo_predict(examples, mrc_model, topk):
    topk_merged_pororo_predictions = []
    all_pororo_preds = []

    for bundle_start_index in tqdm(range(0, len(examples), topk)):
        for example_index in range(bundle_start_index, bundle_start_index + topk):
            example = examples[example_index]

            pororo_pred_text, _, pororo_score = \
                mrc_model(example['question'], example['context'], postprocess=False)[
                    0]
            pororo_pred_text = remove_last_postposition(pororo_pred_text)
            pororo_prediction = {"text": pororo_pred_text, "score": pororo_score}

            topk_merged_pororo_predictions.append(pororo_prediction)

        pororo_pred = max(topk_merged_pororo_predictions,
                          key=lambda x: x["score"])  # 각 context의 top-1 중에서도 top-1만을 추출
        all_pororo_preds.append(pororo_pred)
        topk_merged_pororo_predictions = []

    return all_pororo_preds


def pororo_voting(examples, all_pororo_preds, output_dir, prefix, topk):
    all_pororo_voted_predictions = collections.OrderedDict()
    all_pororo_voted_nbest_json = collections.OrderedDict()

    _, all_nbests = load_predictions_from_json(output_dir, prefix) # len(dataset)
    all_nbests = [val for val in all_nbests.values()] # len(dataset)

    for i, nbest in enumerate(all_nbests):
        example = examples[i * topk]
        for pred in nbest:
            if pred["text"] == all_pororo_preds[i]["text"]:
                pred["score"] += all_pororo_preds[i]["score"] * alpha
                pred["pororo_voting"] = True
        pororo_voted_predictions = sorted(nbest, key=lambda x: x["score"], reverse=True)
        all_pororo_voted_predictions[example["id"]] = pororo_voted_predictions[0]["text"]
        all_pororo_voted_nbest_json[example["id"]] = [
            {k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v) for k, v in
             pred.items()}
            for pred in pororo_voted_predictions
        ]

    return all_pororo_voted_predictions, all_pororo_voted_nbest_json


def pororo_ensemble(examples, output_dir, prefix, topk):
    # PORORO Reader for Voting
    my_mrc_factory = PororoMrcFactory('mrc', 'ko', "brainbert.base.ko.korquad")
    pororo_mrc = my_mrc_factory.load(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    # MRC 모델로 예측하기
    all_pororo_preds = pororo_predict(examples, pororo_mrc, topk)
    return all_pororo_preds
    # MRC 모델 예측 결과를 기존 결과에 합치기
    all_pororo_voted_predictions, all_pororo_voted_nbest_json = pororo_voting(all_pororo_preds, output_dir, prefix, topk)

    # 저장하기
    pororo_voted_prediction_file = os.path.join(
        output_dir, "pororo_predictions.json" if prefix is None else f"pororo_predictions_{prefix}.json"
    )
    pororo_voted_nbest_file = os.path.join(
        output_dir,
        "nbest_pororo_predictions.json" if prefix is None else f"nbest_pororo_predictions_{prefix}.json"
    )
    with open(pororo_voted_prediction_file, "w") as writer:
        writer.write(json.dumps(all_pororo_voted_predictions, indent=4, ensure_ascii=False) + "\n")
    with open(pororo_voted_nbest_file, "w") as writer:
        writer.write(json.dumps(all_pororo_voted_nbest_json, indent=4, ensure_ascii=False) + "\n")

    return all_pororo_voted_predictions



'''
postprocess_qa_predictions
- function 1
    여기서 return하도록.
        - logits : 3중리스트. [ <-- 모델의 600개 prediction을 묶는
                                [ <-- 하나의 question을 묶는  
                                    [(s1, e1), (s2, e2), ...] <-- 하나의 context을 묶는)
                                ] 
                            ]
        - offsets : 마찬가지
        - contexts : 그냥 context 넘겨주세
- function 2 
- function 3 (prediction.json 생성)


- prelimed predictions(->logits), example의 document_id, context, question id(mrc-xxx)값 넘겨주기. 리스트형태로.

- 뽀로로 training args 받아서 사용 하고 말고 결 
'''



def postprocess_qa_predictions(
        examples,
        features,
        predictions: Tuple[np.ndarray, np.ndarray],
        training_args,
        topk: int = 1,
        n_best_size: int = 5,
        max_answer_length: int = 30,
        output_dir: Optional[str] = None,
        prefix: Optional[str] = None,
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
    # predictions,features length check
    all_start_logits, all_end_logits = check_predictions_and_features(predictions, features)

    # one example - n features map
    features_per_example = build_features_per_example_map(examples, topk, features)

    all_prelim_predictions = get_all_prelim_predictions(examples, features, features_per_example,
                                                     all_start_logits, all_end_logits,
                                                     max_answer_length, topk, n_best_size)

    # -> ensemble.py
    if training_args.do_ensemble :
        return all_prelim_predictions, list(examples['context_id']),  list(examples['context']), list(examples['id'])


    all_preds = make_predictions(examples, all_prelim_predictions, topk)

    final_predictions, all_nbest_json = select_top_score_predict(examples, all_preds, n_best_size, topk)

    if output_dir is not None:
        save_predictions_to_json(final_predictions, all_nbest_json, output_dir, prefix)
        if training_args.pororo_prediction:
            all_pororo_voted_predictions = pororo_ensemble()
            return (final_predictions, all_pororo_voted_predictions)

    return (final_predictions)
