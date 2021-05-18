from collections import defaultdict

import numpy as np

from tools import get_args
from evaluation import evaluation
from prepare import get_retriever, get_reader, get_dataset


def offset_visualize():
    pass


def span_visualize():
    pass


def update_offsets(start_scores, end_scores, logits):
    for logit in logits:
        start_scores[logit["offsets"][0]] += logit["start_logit"]
        end_scores[logit["offsets"][1]] += logit["end_logit"]  # pred["text"] = context[offsets[0] : offsets[1]]


def update_spans(span_scores, logits):
    for logit in logits:
        span_scores[logit["offsets"][0] : logit["offsets"][1]] += logit["start_logit"] + logit["end_logit"]  # broadcast


def soft_voting_use_offset(predictions, logits, contexts, document_ids, question_ids):
    for logit, context, doc_id, que_id in zip(logits, contexts, document_ids, question_ids):
        if que_id not in predictions:
            predictions[que_id] = dict()

        if doc_id not in predictions[que_id]:
            predictions[que_id][doc_id] = dict()
            predictions[que_id][doc_id]["sp"] = np.zeros(len(context) + 1)
            predictions[que_id][doc_id]["ep"] = np.zeros(len(context) + 1)
            predictions[que_id][doc_id]["context"] = context

        start_scores = predictions[que_id][doc_id]["sp"]
        end_scores = predictions[que_id][doc_id]["ep"]
        update_offsets(start_scores, end_scores, logit)

        print(f"질문 ID: {que_id}, 문서 ID: {doc_id}")


def soft_voting_use_span(predictions, logits, contexts, document_ids, question_ids):
    for logit, context, doc_id, que_id in zip(logits, contexts, document_ids, question_ids):
        if que_id not in predictions:
            predictions[que_id] = dict()

        if doc_id not in predictions[que_id]:
            predictions[que_id][doc_id] = dict()
            predictions[que_id][doc_id]["span"] = np.zeros(len(context) + 1)
            predictions[que_id][doc_id]["context"] = context

        span_scores = predictions[que_id][doc_id]["span"]
        update_offsets(span_scores, logit)


def hard_voting(logits, offsets, contexts, question_ids):
    pass


def run(args, models, eval_answers, datasets):
    predictions = defaultdict(dict)

    # soft_voting_use_offset

    for model_path in models:
        args.model_name_or_path = model_path

        reader = get_reader(args, eval_answers=eval_answers)
        reader.set_dataset(eval_dataset=datasets["validation"])

        trainer = reader.get_trainer()

        logits, (contexts, document_ids, question_ids) = trainer.get_logits_with_keys(
            reader.eval_dataset, datasets["validation"], keys=["context", "context_id", "id"]
        )

        soft_voting_use_offset(predictions, logits, contexts, document_ids, question_ids)

    #

    print(len(logits), len(contexts), len(document_ids), len(question_ids))
    print(logits[0])
    print(contexts[0])
    print(document_ids[0])
    print(question_ids[0])

    if args.voting == "hard":
        hard_voting(predictions, logits, contexts, document_ids, question_ids)
    elif args.voting == "soft":
        soft_voting_use_offset(predictions, logits, contexts, document_ids, question_ids)

    print(predictions)


def model_ensemble(args):

    MODELS = ["../input/checkpoint/ST01_base_95/checkpoint-1100"]
    args.retriever.topk = 3

    args.retriever.model_name = "BM25"
    args.train.do_predict = True
    args.voting = "soft"

    datasets = get_dataset(args, is_train=False)
    retriever = get_retriever(args)

    eval_answers = datasets["validation"]
    datasets["validation"] = retriever.retrieve(datasets["validation"], topk=args.retriever.topk)["validation"]

    run(args, MODELS, eval_answers, datasets)

    # ENSEMBLE

    # ENSEMBLE PREDICTIONS
    ensemble_result = {}

    for que_id in predictions.keys():

        # (1) Pick, DOC ID
        used_doc = None
        best_score = float("-inf")

        # (2) Start Logits
        for doc_id in predictions[que_id].keys():
            max_score = predictions[que_id][doc_id]["sp"].max()

            if best_score < max_score:
                best_score = max_score
                used_doc = doc_id

        # (2)
        s_offset, e_offset = None, None

        # (s_offset, e_offset) 짝을 이룰 수 있다는 것이 보장되어 있다.
        s_offset = predictions[que_id][used_doc]["sp"].argmax()

        e_offset_start = s_offset + 1
        e_offset_end = e_offset_start + args.data.max_answer_length + 1

        e_offset = e_offset_start + predictions[que_id][used_doc]["ep"][e_offset_start:e_offset_end].argmax()

        ensemble_result[que_id] = predictions[que_id][used_doc]["context"][s_offset:e_offset]

    output_dir = "."
    filename = "test.json"


if __name__ == "__main__":
    #  logit = [
    #      {"offsets": (205, 222), "score": -14.062161, "start_logit": -7.3055415, "end_logit": -6.75662},
    #      {"offsets": (205, 207), "score": -15.824493, "start_logit": -7.3055415, "end_logit": -8.518951},
    #      {"offsets": (219, 222), "score": -14.192354, "start_logit": -7.4357343, "end_logit": -6.75662},
    #  ]
    #
    #  context = "유수일 주교의 문장은 방탄모와 물고기, 비둘기로 육해공군을 형상화해 군종교구장이라는 특수사목직을 명료하게 드러냈다. 성경의 의미가 깊은 상징물을 집약해 인류 구원을 위한 노아의 방주를 중심개념으로 삼았다.\n\n방주의 돛대인 십자가는 하느님과 일치를 이루는 중심축이다. 십자가 3개 점은 그리스도가 십자가에서 흘린 피와 삼위일체 하느님을 상징한다.\n\n비둘기는 노아가 방주에서 날려보낸 비둘기가 올리브 잎을 물고 온 것(창세 8)을 형상화한 것으로, 평화에 대한 희망과 회복의 표현이다. 예수가 제자들에게 '내 평화를 너희에게 준다. 내가 주는 평화는 세상이 주는 평화와 같지 않다.'(요한 14,27)라고 말한 것처럼 군이 사랑과 정의에 바탕을 두고 조국 평화통일을 위해 힘쓰고 있는 점을 부각시켰다.\n\n물고기는 그리스도 신자와 구원받아야 할 영혼을 상징한다. 뒤집어 놓은 방탄모는 지상의 모든 생명을 구원할 방주를 연상시킨다.\n\n문장 하단에는 유수일 주교의 사목표어인 '끊임없이 기도하며'(1테살 5,17)를 넣었다. 영성생활과 복음전파 과업에서 가장 중요한 영적 양식이 기도생활임을 사도 바오로의 테살로니카 신자들에게 보낸 첫째 서간을 통해 드러낸 것이다."
    #  context_id = "mrc-"
    #
    #  start_scores, end_scores = np.zeros(len(context)), np.zeros(len(context))
    #  update_offsets(start_scores, end_scores, logit)

    args = get_args()
    model_ensemble(args)
