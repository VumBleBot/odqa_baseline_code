import tqdm
import time
import os.path as p
from itertools import chain

import torch
import numpy as np
import torch.nn.functional as F
from datasets import load_from_disk, load_dataset
from transformers import TrainingArguments
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from retrieval.dense import DenseRetrieval


def get_retriever_dataset(args):
    if args.retriever.dense_train_dataset not in [
        "train_dataset",
        "squad_kor_v1",
        "bm25_document_questions",
        "bm25_question_documents",
    ]:
        raise FileNotFoundError(f"{args.retriever.dense_train_dataset}은 DenseRetrieval 데이터셋이 아닙니다.")

    if args.retriever.dense_train_dataset == "squad_kor_v1":
        train_dataset = load_dataset(args.retriever.dense_train_dataset)
    else:
        dataset_path = p.join(args.path.train_data_dir, args.retriever.dense_train_dataset)
        assert p.exists(dataset_path), f"{args.retriever.dense_train_dataset}이 경로에 존재하지 않습니다."
        train_dataset = load_from_disk(dataset_path)

    return train_dataset


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


class DprRetrieval(DenseRetrieval):
    def _exec_embedding(self):
        p_encoder, q_encoder = self._load_model()

        train_dataset, eval_dataset = self._load_dataset(eval=True)

        args = TrainingArguments(
            output_dir="dense_retrieval",
            evaluation_strategy="epoch",
            learning_rate=self.args.retriever.learning_rate,
            per_device_train_batch_size=self.args.retriever.per_device_train_batch_size,
            per_device_eval_batch_size=self.args.retriever.per_device_eval_batch_size,
            num_train_epochs=self.args.retriever.num_train_epochs,
            weight_decay=self.args.retriever.weight_decay,
            gradient_accumulation_steps=self.args.retriever.gradient_accumulation_steps,
        )

        p_encoder, q_encoder = self._train(args, train_dataset, p_encoder, q_encoder, eval_dataset)
        p_embedding = []

        for passage in tqdm.tqdm(self.contexts):  # wiki
            passage = self.tokenizer(
                passage, padding="max_length", truncation=True, max_length=512, return_tensors="pt"
            ).to("cuda")
            p_emb = p_encoder(**passage).to("cpu").detach().numpy()
            p_embedding.append(p_emb)

        p_embedding = np.array(p_embedding).squeeze()  # numpy
        return p_embedding, q_encoder


class BaseTrainMixin:
    def _load_dataset(self, eval=False):
        # dataset.features : ['question', 'context', 'answers', ...]
        datasets = get_retriever_dataset(self.args)

        #        tokenizer_input = self.tokenizer(datasets["train"][1]["context"], padding="max_length", max_length=512, truncation=True)
        #        print("tokenizer:", self.tokenizer.convert_ids_to_tokens(tokenizer_input["input_ids"]))

        train_dataset = datasets["train"]

        q_seqs = self.tokenizer(
            train_dataset["question"], padding="longest", truncation=True, max_length=512, return_tensors="pt"
        )
        p_seqs = self.tokenizer(
            train_dataset["context"], padding="max_length", truncation=True, max_length=512, return_tensors="pt"
        )

        train_dataset = TensorDataset(
            p_seqs["input_ids"],
            p_seqs["attention_mask"],
            p_seqs["token_type_ids"],
            q_seqs["input_ids"],
            q_seqs["attention_mask"],
            q_seqs["token_type_ids"],
        )
        eval_dataset = None

        if eval:

            eval_dataset = datasets["validation"]

            q_seqs = self.tokenizer(
                eval_dataset["question"], padding="longest", truncation=True, max_length=512, return_tensors="pt"
            )
            p_seqs = self.tokenizer(
                eval_dataset["context"], padding="max_length", truncation=True, max_length=512, return_tensors="pt"
            )

            eval_dataset = TensorDataset(
                p_seqs["input_ids"],
                p_seqs["attention_mask"],
                p_seqs["token_type_ids"],
                q_seqs["input_ids"],
                q_seqs["attention_mask"],
                q_seqs["token_type_ids"],
            )

        return train_dataset, eval_dataset

    def _train(self, training_args, train_dataset, p_model, q_model, eval_dataset):
        print("TRAINING IN BASE TRAIN MIXIN")
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler, batch_size=training_args.per_device_train_batch_size, drop_last=True
        )
        if eval_dataset:
            eval_sampler = RandomSampler(eval_dataset)
            eval_dataloader = DataLoader(
                eval_dataset, sampler=eval_sampler, batch_size=training_args.per_device_eval_batch_size
            )

        optimizer_grouped_parameters = [{"params": p_model.parameters()}, {"params": q_model.parameters()}]
        optimizer = AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate, eps=training_args.adam_epsilon)

        t_total = len(train_dataloader) // training_args.gradient_accumulation_steps * training_args.num_train_epochs

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=training_args.warmup_steps, num_training_steps=t_total
        )

        global_step = 0

        p_model.train()
        q_model.train()

        p_model.zero_grad()
        q_model.zero_grad()

        torch.cuda.empty_cache()

        for epoch in range(training_args.num_train_epochs):
            train_loss = 0.0
            start_time = time.time()

            for step, batch in enumerate(train_dataloader):

                if torch.cuda.is_available():
                    batch = tuple(t.cuda() for t in batch)

                p_inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2]}
                q_inputs = {"input_ids": batch[3], "attention_mask": batch[4], "token_type_ids": batch[5]}

                p_outputs = p_model(**p_inputs)
                q_outputs = q_model(**q_inputs)

                sim_scores = torch.matmul(q_outputs, torch.transpose(p_outputs, 0, 1))
                targets = torch.arange(0, training_args.per_device_train_batch_size).long()

                if torch.cuda.is_available():
                    targets = targets.to("cuda")

                sim_scores = F.log_softmax(sim_scores, dim=1)
                loss = F.nll_loss(sim_scores, targets)

                loss = loss / training_args.gradient_accumulation_steps

                print(f"epoch: {epoch + 1:02} step: {step:02} loss: {loss}", end="\r")
                train_loss += loss.item()

                loss.backward()

                if ((step + 1) % training_args.gradient_accumulation_steps) == 0:
                    optimizer.step()
                    scheduler.step()

                    p_model.zero_grad()
                    q_model.zero_grad()

                global_step += 1
                torch.cuda.empty_cache()

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            print(f"Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s")
            print(f"\tTrain Loss: {train_loss / len(train_dataloader):.4f}")

            if eval_dataset:
                eval_loss = 0
                correct = 0
                total = 0

                p_model.eval()
                q_model.eval()

                with torch.no_grad():
                    for idx, batch in enumerate(eval_dataloader):
                        if torch.cuda.is_available():
                            batch = tuple(t.cuda() for t in batch)

                        p_inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2]}
                        q_inputs = {"input_ids": batch[3], "attention_mask": batch[4], "token_type_ids": batch[5]}

                        p_outputs = p_model(**p_inputs)
                        q_outputs = q_model(**q_inputs)

                        sim_scores = torch.matmul(q_outputs, torch.transpose(p_outputs, 0, 1))
                        targets = torch.arange(0, training_args.per_device_eval_batch_size).long()

                        if torch.cuda.is_available():
                            targets = targets.to("cuda")

                        sim_scores = F.log_softmax(sim_scores, dim=1)

                        loss = F.nll_loss(sim_scores, targets)

                        loss = loss / training_args.gradient_accumulation_steps

                        predicts = np.argmax(sim_scores.cpu(), axis=1)

                        for idx, predict in enumerate(predicts):
                            total += 1
                            if predict == idx:
                                correct += 1

                        eval_loss += loss.item()

                    print(
                        f"Epoch: {epoch + 1:02}\tEval Loss: {eval_loss / len(eval_dataloader):.4f}\tAccuracy: {correct/total:.4f}"
                    )

            p_model.train()
            q_model.train()

        return p_model, q_model


class Bm25TrainMixin:
    def _load_dataset(self):
        # dataset.features : ['query', 'negative_samples', 'label']
        dataset = get_retriever_dataset(self.args)

        corpus_size = len(dataset["negative_samples"][0])
        negative_samples = list(chain(*dataset["negative_samples"]))

        # query
        q_seqs = self.tokenizer(
            dataset["query"], padding="longest", truncation=True, max_length=512, return_tensors="pt"
        )

        print("query tokenized:", self.tokenizer.convert_ids_to_tokens(q_seqs["input_ids"][0]))

        # negative_samples
        p_seqs = self.tokenizer(
            negative_samples, padding="longest", truncation=True, max_length=512, return_tensors="pt"
        )

        print("negative_sample tokenized:", self.tokenizer.convert_ids_to_tokens(p_seqs["input_ids"][0]))

        embedding_size = p_seqs["input_ids"].shape[-1]

        for k in p_seqs.keys():
            p_seqs[k] = p_seqs[k].reshape(-1, corpus_size, embedding_size)

        train_dataset = TensorDataset(
            p_seqs["input_ids"],
            p_seqs["attention_mask"],
            p_seqs["token_type_ids"],
            q_seqs["input_ids"],
            q_seqs["attention_mask"],
            q_seqs["token_type_ids"],
            torch.tensor(dataset["label"]),
        )

        return train_dataset

    def _train(self, training_args, dataset, p_model, q_model):
        """ Sampling된 데이터 셋으로 학습 """
        print("TRAINING IN BM25 TRAIN MIXIN")

        train_sampler = RandomSampler(dataset)
        train_dataloader = DataLoader(
            dataset, sampler=train_sampler, batch_size=training_args.per_device_train_batch_size
        )

        optimizer_grouped_parameters = [{"params": p_model.parameters()}, {"params": q_model.parameters()}]
        optimizer = AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate, eps=training_args.adam_epsilon)

        t_total = len(train_dataloader) // training_args.gradient_accumulation_steps * training_args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=training_args.warmup_steps, num_training_steps=t_total
        )

        global_step = 0

        p_model.train()
        q_model.train()

        p_model.zero_grad()
        q_model.zero_grad()

        torch.cuda.empty_cache()

        for epoch in range(training_args.num_train_epochs):
            train_loss = 0.0
            start_time = time.time()

            for step, batch in enumerate(train_dataloader):

                if torch.cuda.is_available():
                    batch = tuple(t.cuda() for t in batch)

                # query
                p_inputs = {
                    "input_ids": batch[0].squeeze(),
                    "attention_mask": batch[1].squeeze(),
                    "token_type_ids": batch[2].squeeze(),
                }
                # context
                q_inputs = {"input_ids": batch[3], "attention_mask": batch[4], "token_type_ids": batch[5]}

                label = batch[6]

                p_outputs = p_model(**p_inputs)
                q_outputs = q_model(**q_inputs)

                sim_scores = torch.matmul(q_outputs, torch.transpose(p_outputs, 0, 1))
                sim_scores = F.log_softmax(sim_scores, dim=1)

                loss = F.nll_loss(sim_scores, label) / training_args.gradient_accumulation_steps

                print(f"epoch: {epoch:02} step: {step:02} loss: {loss}", end="\r")
                train_loss += loss.item()

                loss.backward()

                if ((step + 1) % training_args.gradient_accumulation_steps) == 0:
                    optimizer.step()
                    scheduler.step()

                    p_model.zero_grad()
                    q_model.zero_grad()

                global_step += 1

                torch.cuda.empty_cache()

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            print(f"Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s")
            print(f"\tTrain Loss: {train_loss / len(train_dataloader):.4f}")

        # q_model이 document를 encoding했으므로
        if self.args.retriever.dense_train_dataset == "bm25_document_questions":
            return q_model, p_model

        return p_model, q_model
