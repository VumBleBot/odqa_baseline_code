import wandb
import os.path as p
from itertools import product
from argparse import Namespace
from transformers import set_seed
from urllib.error import HTTPError
from tools import update_args, run_test
from trainer_qa import QuestionAnsweringTrainer
from transformers import DataCollatorWithPadding
from prepare import prepare_dataset, preprocess_dataset, get_reader_model, compute_metrics
import unittest
import random
import requests
from copy import deepcopy

def test_train_reader(anc_args):
    args = deepcopy(anc_args)
    strategis = args.strategis
    SEED = random.choice(args.seeds) # fix run_cnt 1

    @run_test
    class TestReader(unittest.TestCase):
        def test_all_strategy(self,args=args):
            self.assertIsNotNone(strategis, "전달받은 전략이 없습니다.")

            for seed,strategy in [(SEED,strategy) for strategy in strategis]:
                try : 
                    update_args(args,strategy)
                except FileNotFoundError:
                    assert False, "전략명이 맞는지 확인해주세요. "
                    
                wandb.init(project="p-stage-3-test", reinit=True)
                args = update_args(args, strategy)  
                args.strategy, args.seed = strategy, seed
                set_seed(seed) 

                try : 
                    prepare_dataset(args, is_train=True, debug=True)
                except KeyError:
                    assert False, "존재하지 않는 dataset입니다. "

                try : 
                    get_reader_model(args)
                except Exception:
                    assert False, "hugging face에 존재하지 않는 model 혹은 잘못된 경로입니다. "

                datasets = prepare_dataset(args, is_train=True) # split train dataset in 5 percent
                model, tokenizer = get_reader_model(args)
                train_dataset, post_processing_function = preprocess_dataset(args, datasets, tokenizer, is_train=True)

                data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if args.train.fp16 else None)

                args.train.do_train = True
                args.train.run_name = "_".join([strategy, str(seed), args.alias, 'test'])
                wandb.run.name = args.train.run_name
                args.train.output_dir = p.join(args.path.checkpoint, args.train.run_name)
                print("checkpoint_dir: ", args.train.output_dir)

                # TRAIN MRC
                args.train.num_train_epochs=1.0 # fix epoch 1
                trainer = QuestionAnsweringTrainer(
                    model=model,
                    args=args.train,  # training_args
                    custom_args=args,
                    train_dataset=train_dataset,
                    tokenizer=tokenizer,
                    data_collator=data_collator,
                    post_process_function=post_processing_function,
                    compute_metrics=compute_metrics,
                )

                trainer.train()
