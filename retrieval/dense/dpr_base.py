import time
import tqdm

import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from transformers import AdamW, TrainingArguments, get_linear_schedule_with_warmup

from retrieval.dense import DenseRetrieval


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


class DprRetrieval(DenseRetrieval):
    def train(self, training_args, dataset, p_model, q_model):
        """ Sampling된 데이터 셋으로 학습 """

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

                p_inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2]}
                q_inputs = {"input_ids": batch[3], "attention_mask": batch[4], "token_type_ids": batch[5]}
                label = batch[6]

                p_outputs = p_model(**p_inputs)
                q_outputs = q_model(**q_inputs)

                sim_scores = torch.matmul(q_outputs, torch.transpose(p_outputs, 0, 1))
                sim_scores = F.log_softmax(sim_scores, dim=1)

                loss = F.nll_loss(sim_scores, label)

                print(f"epoch: {epoch:02} step: {step:02} loss: {loss}", end="\r")
                train_loss += loss.item()

                loss.backward()
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

        return p_model, q_model

    def _exec_embedding(self):
        p_encoder, q_encoder = self._load_model()

        train_dataset = self._load_dataset()

        args = TrainingArguments(
            output_dir="dense_retrieval",
            evaluation_strategy="epoch",
            learning_rate=self.args.retriever.learning_rate,
            per_device_train_batch_size=self.args.retriever.per_device_train_batch_size,
            per_device_eval_batch_size=self.args.retriever.per_device_eval_batch_size,
            num_train_epochs=self.args.retriever.num_train_epochs,
            weight_decay=self.args.retriever.weight_decay,
        )

        p_encoder, q_encoder = self.train(args, train_dataset, p_encoder, q_encoder)

        p_embedding = []

        for passage in tqdm.tqdm(self.contexts):  # wiki
            passage = self.tokenizer(
                passage, padding="max_length", truncation=True, max_length=512, return_tensors="pt"
            ).to("cuda")
            p_emb = p_encoder(**passage).to("cpu").detach().numpy()
            p_embedding.append(p_emb)

        p_embedding = np.array(p_embedding).squeeze()  # numpy
        return p_embedding, q_encoder
