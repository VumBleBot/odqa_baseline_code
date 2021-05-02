import os.path as p

from tools import update_args
from trainer_qa import QuestionAnsweringTrainer
from transformers import DataCollatorWithPadding
from prepare import prepare_dataset, preprocess_dataset, get_reader_model, compute_metrics, get_retriever


def predict(args):
    # Don't use wandb

    strategis = args.strategis

    for idx, strategy in enumerate(strategis):
        args = update_args(args, strategy)  # auto add args.save_path, args.base_path
        args.strategy = strategy
        args.train.do_predict = True
        args.train.output_dir = p.join(args.path.checkpoint, strategy)

        datasets = prepare_dataset(args, is_train=False)
        model, tokenizer = get_reader_model(args)

        retriever = get_retriever(args)
        retriever.get_sparse_embedding()

        datasets = retriever.retrieve_pipeline(args, datasets["validation"])
        eval_dataset, post_processing_function = preprocess_dataset(args, datasets, tokenizer, is_train=False)

        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if args.train.fp16 else None)

        trainer = QuestionAnsweringTrainer(
            model=model,
            args=args.train,  # training_args
            custom_args=args,
            train_dataset=None,
            eval_dataset=eval_dataset,
            eval_examples=datasets["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            post_process_function=post_processing_function,
            compute_metrics=compute_metrics,
        )

        trainer.predict(test_dataset=eval_dataset, test_examples=datasets["validation"])


if __name__ == "__main__":
    from tools import get_args

    args = get_args()

    predict(args)
