""" Ensemble을 수행하는 코드입니다. """

import json
import os.path as p
from collections import defaultdict

import numpy as np
from tqdm.auto import tqdm

from utils.tools import get_args, update_args
from utils.prepare import get_retriever, get_reader, get_dataset

TOPK = 5
MAX_ANSWER_LENGTH = 30
OFFSET_DEFAULT = 0
SPAN_DEFAULT = 0


def postprocess(predictions, key="sp"):
    """ 0이 아닌 값들의 최소값을 1로 맞춘다. """

    min_value_list = []

    for que_id in predictions.keys():
        for doc_id in predictions[que_id].keys():
            doc_min_score = predictions[que_id][doc_id][key].min()
            min_value_list.append(doc_min_score)

    best_min = min(min_value_list) + 1

    for que_id in predictions.keys():
        for doc_id in predictions[que_id].keys():
            f_idxs = np.where(predictions[que_id][doc_id][key] != OFFSET_DEFAULT)
            predictions[que_id][doc_id][key][f_idxs] += best_min


def offset_postprocess(predictions):
    postprocess(predictions, key="sp")
    postprocess(predictions, key="ep")


def span_postprocess(predictions):
    postprocess(predictions, key="span")


def logit_list_standardization(logit_list):
    start_logits, end_logits = [], []

    for logits in logit_list:
        for logit in logits:
            start_logits.append(logit["start_logit"])
            end_logits.append(logit["end_logit"])

    start_logits = np.array(start_logits)
    end_logits = np.array(end_logits)

    for logits in logit_list:
        for logit in logits:
            logit["start_logit"] = (logit["start_logit"] - start_logits.mean()) / start_logits.std()
            logit["end_logit"] = (logit["end_logit"] - end_logits.mean()) / end_logits.std()

    return logit_list


def update_hard_offsets(start_scores, end_scores, logits):
    for logit in logits:
        start_scores[logit["offsets"][0]] = max(start_scores[logit["offsets"][0]], logit["start_logit"])
        end_scores[logit["offsets"][1]] = max(start_scores[logit["offsets"][1]], logit["end_logit"])


def update_soft_offsets(start_scores, end_scores, logits):
    for logit in logits:
        start_scores[logit["offsets"][0]] += logit["start_logit"]
        end_scores[logit["offsets"][1]] += logit["end_logit"]  # pred["text"] = context[offsets[0] : offsets[1]]


def update_spans(span_scores, logits):
    for logit in logits:
        span_scores[logit["offsets"][0] : logit["offsets"][1]] += logit["start_logit"] + logit["end_logit"]  # broadcast


def soft_voting_use_offset(predictions, logits, contexts, document_ids, question_ids):
    for logit, context, doc_id, que_id in tqdm(
        zip(logits, contexts, document_ids, question_ids), desc="Soft Voting Use Offset"
    ):
        if que_id not in predictions:
            predictions[que_id] = dict()

        if doc_id not in predictions[que_id]:
            predictions[que_id][doc_id] = dict()
            predictions[que_id][doc_id]["sp"] = np.zeros(len(context) + 1) + OFFSET_DEFAULT
            predictions[que_id][doc_id]["ep"] = np.zeros(len(context) + 1) + OFFSET_DEFAULT
            predictions[que_id][doc_id]["context"] = context

        start_scores = predictions[que_id][doc_id]["sp"]
        end_scores = predictions[que_id][doc_id]["ep"]
        update_soft_offsets(start_scores, end_scores, logit)


def hard_voting_use_offset(predictions, logits, contexts, document_ids, question_ids):
    for logit, context, doc_id, que_id in tqdm(
        zip(logits, contexts, document_ids, question_ids), desc="Soft Voting Use Offset"
    ):
        if que_id not in predictions:
            predictions[que_id] = dict()

        if doc_id not in predictions[que_id]:
            predictions[que_id][doc_id] = dict()
            predictions[que_id][doc_id]["sp"] = np.zeros(len(context) + 1) + OFFSET_DEFAULT
            predictions[que_id][doc_id]["ep"] = np.zeros(len(context) + 1) + OFFSET_DEFAULT
            predictions[que_id][doc_id]["context"] = context

        start_scores = predictions[que_id][doc_id]["sp"]
        end_scores = predictions[que_id][doc_id]["ep"]
        update_hard_offsets(start_scores, end_scores, logit)


def soft_voting_use_span(predictions, logits, contexts, document_ids, question_ids):
    for logit, context, doc_id, que_id in tqdm(
        zip(logits, contexts, document_ids, question_ids), desc="Soft Voting Use Span"
    ):
        if que_id not in predictions:
            predictions[que_id] = dict()

        if doc_id not in predictions[que_id]:
            predictions[que_id][doc_id] = dict()
            predictions[que_id][doc_id]["span"] = np.zeros(len(context) + 1) + SPAN_DEFAULT
            predictions[que_id][doc_id]["context"] = context

        span_scores = predictions[que_id][doc_id]["span"]
        update_spans(span_scores, logit)


def save_offset_ensemble(args, predictions, filename):
    ensemble_results = {}

    for que_id in predictions.keys():
        used_doc = None
        best_score = float("-inf")

        for doc_id in predictions[que_id].keys():
            max_score = predictions[que_id][doc_id]["sp"].max()

            if best_score < max_score:
                best_score = max_score
                used_doc = doc_id

        s_offset, e_offset = None, None
        s_offset = predictions[que_id][used_doc]["sp"].argmax()

        e_offset_start = s_offset + 1
        e_offset_end = e_offset_start + args.data.max_answer_length + 1

        e_offset = e_offset_start + predictions[que_id][used_doc]["ep"][e_offset_start:e_offset_end].argmax()
        ensemble_results[que_id] = predictions[que_id][used_doc]["context"][s_offset:e_offset]

    save_path = p.join(args.path.info, filename)

    with open(save_path, "w") as f:
        f.write(json.dumps(ensemble_results, indent=4, ensure_ascii=False) + "\n")


def save_span_ensemble(args, predictions, filename, percent=75):
    ensemble_results = {}

    for que_id in predictions.keys():
        used_doc = None
        best_score = float("-inf")

        for doc_id in predictions[que_id].keys():
            max_score = predictions[que_id][doc_id]["span"].max()

            if best_score < max_score:
                best_score = max_score
                used_doc = doc_id

        peak = np.argmax(predictions[que_id][used_doc]["span"])
        sample = predictions[que_id][doc_id]["span"][
            max(peak - MAX_ANSWER_LENGTH // 2, 0) : peak + MAX_ANSWER_LENGTH // 2
        ]

        if len(sample) != 0:
            sample_75 = np.percentile(sample, percent)
            sample = np.array(list(map(lambda x: 0 if x < sample_75 else x, sample)))

        sample_index = np.where(sample > 0)
        sample_index = (sample_index[0] + peak - 15,)  # tuple

        s_offset, e_offset = sample_index[0][0], sample_index[0][-1] + 1
        ensemble_results[que_id] = predictions[que_id][used_doc]["context"][s_offset:e_offset]

    save_path = p.join(args.path.info, filename)

    with open(save_path, "w") as f:
        f.write(json.dumps(ensemble_results, indent=4, ensure_ascii=False) + "\n")


def run(args, models, eval_answers, datasets):
    """Ensemble을 수행합니다.
    1. Soft Voting Use Offset
    2. Soft Voting Use Span
    3. Hard Voting Use Offset
    """

    soft_offset_predictions = defaultdict(dict)
    soft_span_predictions = defaultdict(dict)
    hard_offset_predictions = defaultdict(dict)

    for model_path, strategy in models:
        args.model_name_or_path = model_path
        args.model.reader_name = "DPR"

        if strategy is not None:
            args = update_args(args, strategy)

        args.retriever.topk = TOPK

        reader = get_reader(args, eval_answers=eval_answers)
        reader.set_dataset(eval_dataset=datasets["validation"])

        trainer = reader.get_trainer()

        logit_list, (contexts, document_ids, question_ids) = trainer.get_logits_with_keys(
            reader.eval_dataset, datasets["validation"], keys=["context", "context_id", "id"]
        )

        # Logit Standardization, ~1 ~ 1
        logit_list = logit_list_standardization(logit_list)

        soft_voting_use_offset(soft_offset_predictions, logit_list, contexts, document_ids, question_ids)
        hard_voting_use_offset(hard_offset_predictions, logit_list, contexts, document_ids, question_ids)
        soft_voting_use_span(soft_span_predictions, logit_list, contexts, document_ids, question_ids)

    offset_postprocess(soft_offset_predictions)
    offset_postprocess(hard_offset_predictions)
    span_postprocess(soft_span_predictions)

    filename = "soft_offset_predictions.json"
    save_offset_ensemble(args, soft_offset_predictions, filename)

    filename = "hard_offset_predictions.json"
    save_offset_ensemble(args, hard_offset_predictions, filename)

    filename = "soft_span_predictions.json"
    save_span_ensemble(args, hard_offset_predictions, filename)


def model_ensemble(args):
    """ 직접 모델과 전략 입력해주시면 됩니다! """

    MODELS = [
        ("../input/model_ensemble_checkpoint/gunmo/RD_G04_C01_KOELECTRA_BASE_V3_FINETUNED_95/checkpoint-6000/", None),
        (
            "../input/model_ensemble_checkpoint/suyeon/KOELECTRA_FINETUNED_TRAIN_KOELECTRA_FINETUNED_95/checkpoint-5400/",
            None,
        ),
        ("../input/model_ensemble_checkpoint/suyeon/ST05_AtireBM25_95/checkpoint-5000/", None),
        ("../input/model_ensemble_checkpoint/jonghun/ST101_CNN_95/checkpoint-15100/", "ST101"),
        ("../input/model_ensemble_checkpoint/jonghun/ST103_CNN_LSTM_95/checkpoint-5500/", "ST103"),
        ("../input/model_ensemble_checkpoint/jonghun/ST104_CCNN_v2_95/checkpoint-15100/", "ST104"),
        ("../input/model_ensemble_checkpoint/jonghun/ST106_LSTM_95/checkpoint-1500/", "ST106"),
    ]

    args.retriever.topk = TOPK
    args.data.max_answer_length = MAX_ANSWER_LENGTH
    args.retriever.model_name = "ATIREBM25_DPRBERT"
    args.train.do_predict = True

    datasets = get_dataset(args, is_train=False)
    retriever = get_retriever(args)

    eval_answers = datasets["validation"]
    datasets["validation"] = retriever.retrieve(datasets["validation"], topk=args.retriever.topk)["validation"]

    run(args, MODELS, eval_answers, datasets)


if __name__ == "__main__":
    args = get_args()
    model_ensemble(args)
