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
from tools import get_args

args = get_args()

strategis = args.strategis
SEED = random.choice(args.seeds) # fix run_cnt 1

@run_test
class TestReader(unittest.TestCase):
    def test_strategy_is_not_none(self,args=args):
        self.assertIsNotNone(strategis, "전달받은 전략이 없습니다.")
    
    def test_valid_strategy(self,args=args):
        for strategy in strategis:
            try : 
                update_args(args,strategy)
            except FileNotFoundError:
                assert False, "전략명이 맞는지 확인해주세요. "
                
    def test_valid_dataset(self,args=args):
        for seed,strategy in [(SEED,strategy) for strategy in strategis]:
            args = update_args(args, strategy) 
            args.strategy, args.seed = strategy, seed
            set_seed(seed) 
            try : 
                prepare_dataset(args, is_train=True)
            except KeyError:
                assert False, "존재하지 않는 dataset입니다. "

    def test_valid_model(self,args=args):
        for seed,strategy in [(SEED,strategy) for strategy in strategis]:
            args = update_args(args, strategy) 
            args.strategy, args.seed = strategy, seed
            set_seed(seed) 
            try : 
                get_reader_model(args)
            except Exception:
                assert False, "hugging face에 존재하지 않는 model 혹은 잘못된 경로입니다. "

    def test_strategis_with_dataset(self,args=args):
        """
            (Constraint)
                - num_train_epoch 1
                - random seed 1
                - dataset fragment (rows : 100)
            (Caution) 
                ERROR가 표시된다면, 상위 단위 테스트 결과를 확인하세요.
        """
        for seed,strategy in [(SEED,strategy) for strategy in strategis]:
            wandb.init(project="p-stage-3-test", reinit=True)
            args = update_args(args, strategy)  
            args.strategy, args.seed = strategy, seed
            set_seed(seed) 

            datasets = prepare_dataset(args, is_train=True) 
            model, tokenizer = get_reader_model(args)
            train_dataset, post_processing_function = preprocess_dataset(args, datasets, tokenizer, is_train=True)
            
            train_dataset = train_dataset.select(range(100)) # select 100
            
            data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if args.train.fp16 else None)

            args.train.do_train = True
            args.train.run_name = "_".join([strategy, args.alias, str(seed), 'test'])
            wandb.run.name = args.train.run_name

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
