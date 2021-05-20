import os.path as p

from datasets import DatasetDict, Dataset
from sklearn.model_selection import KFold

from tools import update_args
from prepare import get_reader, get_retriever, get_full_dataset, get_dataset


def kfold_train_and_predict(args):
    # Don't use wandb

    strategies = args.strategies

    for idx, strategy in enumerate(strategies):
        args = update_args(args, strategy)  # auto add args.save_path, args.base_path
        args.strategy = strategy

        # args.model.model_name_or_path = args.model_path
        args.train.output_dir = p.join(args.path.checkpoint, strategy)

        kf = KFold(n_splits=args.train.fold_num)
        kfold_predictions = dict()

        datasets_list = []

        full_ds = get_full_dataset(args)
        for index, (tr, val) in enumerate(kf.split(full_ds)):
            print(len(tr), len(val))
            ds = DatasetDict()
            ds['train'] = Dataset.from_dict(full_ds[tr])
            ds['validation'] = Dataset.from_dict(full_ds[val])

            datasets_list.append(ds)

        # test dataset
        test_dataset = get_dataset(args, is_train=False)

        # pororo_prediction은 마지막에만 사용
        args.train.pororo_prediction = False

        for index, datasets in enumerate(datasets_list):
            print(f"FOLD {index + 1} TRAIN DATASET 길이: {len(datasets['train'])}")
            print(f"FOLD {index + 1} VALID DATASET 길이: {len(datasets['validation'])}")

            ### 학습 ###

            # reader 모델 학습시에는 topk=1로 고정
            topk = args.retriever.topk
            args.retriever.topk = 1

            args.train.do_predict = False

            reader = get_reader(args, eval_answers=datasets["validation"])

            reader.set_dataset(train_dataset=datasets["train"], eval_dataset=datasets["validation"])

            trainer = reader.get_trainer()

            # 학습
            train_results = trainer.train()
            print(f"{'*' * 20}FOLD {index} train done.{'*' * 20}")
            metrics = trainer.evaluate()
            print(f"EM : {metrics['exact_match']} | f1 : {metrics['f1']} | EPOCH : {metrics['epoch']}")

            ### 학습 끝 ###

            ### 예측 ###

            # 예측시에 topk 복원
            args.retriever.topk = topk

            args.train.do_predict = True

            reader = get_reader(args, eval_answers=test_dataset["validation"])
            retriever = get_retriever(args)
            retrieved_test_dataset = retriever.retrieve(test_dataset["validation"], topk = args.retriever.topk)
            reader.set_dataset(eval_dataset=retrieved_test_dataset["validation"])

            output = trainer.get_prediction_output('predict', dataset=reader.eval_dataset,
                                                   examples=retrieved_test_dataset)

            print(f"{'*' * 20}FOLD {index + 1} predict done.{'*' * 20}")

            kfold_predictions[index] = output.predictions

            print(f"Fold {index + 1} prediction is merged into kfold prediction.")


        kfold_prediction = None
        for fold_index, pred in kfold_predictions.items():
            if kfold_prediction is None:
                kfold_prediction = list(pred)
            else:
                kfold_prediction[0] += pred[0]
                kfold_prediction[1] += pred[1]

        kfold_prediction[0] /= args.train.fold_num  # mean of all start logits
        kfold_prediction[1] /= args.train.fold_num  # mean of all end logits

        # use pororo_predict at last time
        args.train.pororo_prediction = True
        reader = get_reader(args, eval_answers=test_dataset["validation"])
        retriever = get_retriever(args)
        retrieved_test_dataset = retriever.retrieve(test_dataset["validation"], topk = args.retriever.topk)
        reader.set_dataset(eval_dataset=retrieved_test_dataset["validation"])

        trainer.kfold_predict(test_dataset=reader.eval_dataset, test_examples=retrieved_test_dataset["validation"],
                              kfold_output=tuple(kfold_prediction))



if __name__ == "__main__":
    from tools import get_args, update_args

    args = get_args()

    kfold_train_and_predict(args)
    print("K-fold prediction finished.")
