import tqdm
import time, os
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


class BprRetrieval(DenseRetrieval):

    def get_relevant_doc_bulk(self, queries, topk=1):
        self.encoder.eval()  # question encoder
        self.encoder.cuda()

        with torch.no_grad():
            q_seqs_val = self.tokenizer(
                queries, padding="longest", truncation=True, max_length=512, return_tensors="pt"
            ).to("cuda")
            q_embedding = self.encoder(**q_seqs_val)
            q_embedding.squeeze_()  # in-place
            q_embedding = q_embedding.cpu().detach().numpy()
            q_embedding = self.encoder.convert_to_binary_code(q_embedding)


        # p_embedding: numpy, q_embedding: numpy
        result = np.matmul(q_embedding, self.p_embedding.T)
        phrase_indices = np.argsort(result, axis=1)[:, -topk:][:, ::-1]
        doc_indices = [[self.mappings[phrase_indices[i][j]] for j in range(len(phrase_indices[i]))] for i in
                       range(len(phrase_indices))]
        doc_scores = []

        for i in range(len(phrase_indices)):
            doc_scores.append(result[i][[phrase_indices[i].tolist()]])

        return doc_scores, doc_indices

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
            warmup_ratio=self.args.retriever.warmup_ratio,
        )

        existed_p_dir = self.args.retriever.existed_p_dir
        existed_q_dir = self.args.retriever.existed_q_dir
        skip_epochs = self.args.retriever.skip_epochs

        p_encoder, q_encoder = self._train(args, train_dataset, p_encoder, q_encoder, eval_dataset, existed_p_dir,
                                           existed_q_dir, skip_epochs)
        p_embedding = []

        # passage => phrase
        for passage in tqdm.tqdm(self.contexts):  # wiki
            passage = self.tokenizer(
                passage, padding="max_length", truncation=True, max_length=512, return_tensors="pt"
            ).to("cuda")
            p_emb = p_encoder(**passage).to("cpu").detach().numpy()
            p_emb = p_encoder.convert_to_binary_code(p_emb)
            p_embedding.append(p_emb)

        p_embedding = np.array(p_embedding).squeeze()  # numpy
        return p_embedding, q_encoder

    # Will be replaced by Jiho's
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

    def _train(self, training_args, train_dataset, p_model, q_model, eval_dataset, existed_p_dir, existed_q_dir,
               skip_epochs=0):
        print("TRAINING Binary Passage Retriever")

        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler, batch_size=training_args.per_device_train_batch_size, drop_last=True
        )
        if eval_dataset:
            eval_sampler = RandomSampler(eval_dataset)
            eval_dataloader = DataLoader(
                eval_dataset, sampler=eval_sampler, batch_size=training_args.per_device_eval_batch_size
            )

        if existed_p_dir:
            p_model.load_state_dict(torch.load(existed_p_dir))

        if existed_q_dir:
            q_model.load_state_dict(torch.load(existed_p_dir))

        optimizer_grouped_parameters = [{"params": p_model.parameters()}, {"params": q_model.parameters()}]
        optimizer = AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate, eps=training_args.adam_epsilon)

        t_total = len(train_dataloader) // training_args.gradient_accumulation_steps * training_args.num_train_epochs
        warmup_steps = int(training_args.warmup_ratio * t_total)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        )

        global_step = 0

        p_model.train()
        q_model.train()
        p_model.training = True
        q_model.training = True

        p_model.zero_grad()
        q_model.zero_grad()

        torch.cuda.empty_cache()

        save_dir = p.join(self.save_dir, f"{time.ctime()}")
        if not p.exists(save_dir):
            os.mkdir(save_dir)

        for epoch in range(training_args.num_train_epochs):
            train_loss = 0.0
            start_time = time.time()

            for step, batch in enumerate(train_dataloader):
                # Skip epochs to continue learning
                if epoch < skip_epochs:
                    optimizer.step()
                    scheduler.step()
                    global_step += 1
                    continue

                if torch.cuda.is_available():
                    batch = tuple(t.cuda() for t in batch)

                p_inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2]}
                q_inputs = {"input_ids": batch[3], "attention_mask": batch[4], "token_type_ids": batch[5]}

                p_outputs = p_model(**p_inputs)  # shape = [batch_size, embedding_size]
                q_outputs = q_model(**q_inputs)  # shape = [batch_size, embedding_size]

                # Convert embeddings to binary code
                p_outputs = p_model.convert_to_binary_code(p_outputs, global_step)

                # NOTE: Check this
                targets = torch.arange(0, training_args.per_device_train_batch_size).long()

                if torch.cuda.is_available():
                    targets = targets.to("cuda")

                scores = torch.matmul(q_outputs, p_outputs.transpose(0, 1))
                dense_loss = F.cross_entropy(scores, targets)  # Rerank loss

                binary_q_outputs = q_model.convert_to_binary_code(q_outputs, global_step)
                binary_q_scores = torch.matmul(binary_q_outputs, p_outputs.transpose(0, 1))

                if self.args.retriever.use_binary_cross_entropy_loss:
                    binary_loss = F.cross_entropy(binary_q_scores, targets)
                else:
                    pos_mask = binary_q_scores.new_zeros(binary_q_scores.size(), dtype=torch.bool)
                    for n, label in enumerate(targets):
                        pos_mask[n, label] = True
                    pos_bin_scores = torch.masked_select(binary_q_scores, pos_mask)
                    pos_bin_scores = pos_bin_scores.repeat_interleave(p_outputs.size(0) - 1)
                    neg_bin_scores = torch.masked_select(binary_q_scores, torch.logical_not(pos_mask))
                    bin_labels = pos_bin_scores.new_ones(pos_bin_scores.size(), dtype=torch.int64)
                    binary_loss = F.margin_ranking_loss(
                        pos_bin_scores, neg_bin_scores, bin_labels, self.args.retriever.binary_ranking_loss_margin,
                    )

                loss = binary_loss + dense_loss

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

            # Skip epochs to continue learning
            if epoch < skip_epochs:
                print(f"skipping {epoch} epoch...")
                continue

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            print(f"Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s")
            print(f"\tTrain Loss: {train_loss / len(train_dataloader):.4f}")

            if epoch % 5 == 0:
                print("Save model...")
                torch.save(p_model.state_dict(), p.join(save_dir, f"{self.name}-p.pth"))
                torch.save(q_model.state_dict(), p.join(save_dir, f"{self.name}-q.pth"))
                print("Save Success!")

            if eval_dataset:
                eval_loss = 0
                correct = 0
                total = 0

                p_model.eval()
                q_model.eval()
                p_model.training = False
                q_model.training = False

                with torch.no_grad():
                    for idx, batch in enumerate(eval_dataloader):
                        if torch.cuda.is_available():
                            batch = tuple(t.cuda() for t in batch)

                        p_inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2]}
                        q_inputs = {"input_ids": batch[3], "attention_mask": batch[4], "token_type_ids": batch[5]}

                        p_outputs = p_model(**p_inputs)
                        q_outputs = q_model(**q_inputs)

                        p_outputs = p_model.convert_to_binary_code(p_outputs)

                        # NOTE: Check this
                        targets = torch.arange(0, training_args.per_device_eval_batch_size).long()

                        if torch.cuda.is_available():
                            targets = targets.to("cuda")

                        scores = torch.matmul(q_outputs, p_outputs.transpose(0, 1))
                        dense_loss = F.cross_entropy(scores, targets)  # Rerank loss

                        binary_q_outputs = q_model.convert_to_binary_code(q_outputs)
                        binary_q_scores = torch.matmul(binary_q_outputs, p_outputs.transpose(0, 1))

                        if self.args.retriever.use_binary_cross_entropy_loss:
                            binary_loss = F.cross_entropy(binary_q_scores, targets)
                        else:
                            pos_mask = binary_q_scores.new_zeros(binary_q_scores.size(), dtype=torch.bool)
                            for n, label in enumerate(targets):
                                pos_mask[n, label] = True
                            pos_bin_scores = torch.masked_select(binary_q_scores, pos_mask)
                            pos_bin_scores = pos_bin_scores.repeat_interleave(p_outputs.size(0) - 1)
                            neg_bin_scores = torch.masked_select(binary_q_scores, torch.logical_not(pos_mask))
                            bin_labels = pos_bin_scores.new_ones(pos_bin_scores.size(), dtype=torch.int64)
                            binary_loss = F.margin_ranking_loss(
                                pos_bin_scores, neg_bin_scores, bin_labels,
                                self.args.retriever.binary_ranking_loss_margin,
                            )

                        loss = binary_loss + dense_loss

                        loss = loss / training_args.gradient_accumulation_steps

                        rerank_predicts = np.argmax(scores.cpu(), axis=1)

                        for idx, predict in enumerate(rerank_predicts):
                            total += 1
                            if predict == idx:
                                correct += 1

                        eval_loss += loss.item()

                    print(
                        f"Epoch: {epoch + 1:02}\tEval Loss: {eval_loss / len(eval_dataloader):.4f}\tRerank Accuracy: {correct / total:.4f}"
                    )
                    # print(
                    #     f"Epoch: {epoch + 1:02}\tEval Loss: {eval_loss / len(eval_dataloader):.4f}\tCandidate Accuracy: {loss}\tRerank Accuracy: {correct/total:.4f}"
                    # )

            p_model.train()
            q_model.train()
            p_model.training = True
            q_model.training = True

        return p_model, q_model


