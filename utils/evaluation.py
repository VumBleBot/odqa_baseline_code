# Inspired by KorQuad 1.0 evaluation script.
# https://korquad.github.io/KorQuad%201.0/

import re
import sys
import json
import string
import os.path as p
from collections import Counter

from datasets import load_from_disk


def get_gt_json(args):
    """Get the json file that contain ground truth of validation datasets.
        If not exists, generate the json file.

    Arguments:
        args: user arguments
    Return:
        gt_json
    """
    gt_json = None
    save_path = p.join(args.path.train_data_dir, "eval_gt.json")

    if p.isfile(save_path):
        with open(save_path, "r", encoding="utf-8") as f:
            gt_json = json.load(f)
    else:
        eval_datasets = load_from_disk(p.join(args.path.train_data_dir, args.data.dataset_name))["validation"]

        gt_json = []
        for data in eval_datasets:
            result = {"id": data["id"], "answer": data["answers"]["text"]}
            gt_json.append(result)

        with open(save_path, "w", encoding="utf-8") as save_file:
            json.dump(gt_json, save_file, indent=4, ensure_ascii=False)

    return gt_json


def evaluation(args, prefix=""):
    """Calculate MRC metrics.

    Arguments:
        args: args
    """

    pred_path = p.join(args.train.output_dir, f"{prefix}predictions_valid.json")
    save_path = p.join(args.train.output_dir, f"{prefix}valid_results.json")

    gt = get_gt_json(args)
    with open(pred_path) as pred_file:
        preds = json.load(pred_file)

    f1 = exact_match = total = 0

    for qa in gt:
        total += 1
        if qa["id"] not in preds:
            message = "Unanswered question " + qa["id"] + " will receive score 0."
            print(message, file=sys.stderr)
            continue
        ground_truths = qa["answer"]
        prediction = preds[qa["id"]]
        exact_match += metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)

    exact_match = exact_match / total
    f1 = f1 / total

    results = {}
    results["EM"] = {"value": f"{exact_match:.2%}", "rank": True, "decs": True}
    results["F1"] = {"value": f"{f1:.2%}", "rank": False, "decs": True}

    with open(save_path, "w", encoding="utf-8") as save_file:
        json.dump(results, save_file, indent=4, ensure_ascii=False)

    return results


def normalize_answer(s):
    def remove_(text):
        """ 불필요한 기호 제거 """
        text = re.sub("'", " ", text)
        text = re.sub('"', " ", text)
        text = re.sub("《", " ", text)
        text = re.sub("》", " ", text)
        text = re.sub("<", " ", text)
        text = re.sub(">", " ", text)
        text = re.sub("〈", " ", text)
        text = re.sub("〉", " ", text)
        text = re.sub("\(", " ", text)
        text = re.sub("\)", " ", text)
        text = re.sub("‘", " ", text)
        text = re.sub("’", " ", text)
        return text

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(remove_(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    # F1 by character
    prediction_Char = []
    for tok in prediction_tokens:
        now = [a for a in tok]
        prediction_Char.extend(now)

    ground_truth_Char = []
    for tok in ground_truth_tokens:
        now = [a for a in tok]
        ground_truth_Char.extend(now)

    common = Counter(prediction_Char) & Counter(ground_truth_Char)
    num_same = sum(common.values())
    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(prediction_Char)
    recall = 1.0 * num_same / len(ground_truth_Char)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)
