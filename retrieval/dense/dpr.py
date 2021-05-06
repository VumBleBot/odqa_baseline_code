import tqdm
import os.path as p

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import (
    AutoModel,
    AutoConfig,
    BertConfig,
    BertModel,
    AutoTokenizer,
    BertTokenizer,
    BertPreTrainedModel,
    AdamW,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)

from retrieval.dense import DenseRetrieval
from tokenization_kobert import KoBertTokenizer


class BertEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]  # embedding 가져오기
        return pooled_output


class AutoEncoder(nn.Module):
    def __init__(self, model_name, config):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name, config=config)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.backbone(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]  # embedding 가져오기
        return pooled_output


class DprRetrieval(DenseRetrieval):
    def __init__(self, args):
        super().__init__(args)
        self.backbone = "bert-base-multilingual-cased"
        self.tokenizer = BertTokenizer.from_pretrained(self.backbone)

    def _load_model(self):
        config = BertConfig.from_pretrained(self.backbone)
        p_encoder = BertEncoder.from_pretrained(self.backbone, config=config).cuda()
        q_encoder = BertEncoder.from_pretrained(self.backbone, config=config).cuda()
        return p_encoder, q_encoder

    def _get_encoder(self):
        config = BertConfig.from_pretrained(self.backbone)
        q_encoder = BertEncoder(config=config).cuda()
        return q_encoder

    def train(self, training_args, dataset, p_model, q_model):
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
            total_loss = 0.0

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

                print(f"epoch: {epoch:02} step: {step:02} loss: {loss}", end="\r")
                total_loss += loss.item()

                loss.backward()
                optimizer.step()
                scheduler.step()
                p_model.zero_grad()
                q_model.zero_grad()
                global_step += 1

                torch.cuda.empty_cache()

            print(f"epoch: {epoch:02} step: {step:02} loss: {total_loss / len(train_dataloader)}")

        return p_model, q_model

    def _exec_embedding(self):
        p_encoder, q_encoder = self._load_model()

        datasets = load_from_disk(p.join(self.args.path.train_data_dir, self.args.retriever.dense_train_dataset))
        tokenizer_input = self.tokenizer(datasets["train"][1]["context"], padding="max_length", truncation=True)

        print("tokenizer:", self.tokenizer.convert_ids_to_tokens(tokenizer_input["input_ids"]))

        train_dataset = datasets["train"]

        q_seqs = self.tokenizer(train_dataset["question"], padding="max_length", truncation=True, return_tensors="pt")
        p_seqs = self.tokenizer(train_dataset["context"], padding="max_length", truncation=True, return_tensors="pt")

        train_dataset = TensorDataset(
            p_seqs["input_ids"],
            p_seqs["attention_mask"],
            p_seqs["token_type_ids"],
            q_seqs["input_ids"],
            q_seqs["attention_mask"],
            q_seqs["token_type_ids"],
        )

        args = TrainingArguments(
            output_dir="dense_retrieval",
            evaluation_strategy="epoch",
            learning_rate=3e-5,
            per_device_train_batch_size=1000,
            per_device_eval_batch_size=4,
            num_train_epochs=2,
            weight_decay=0.01,
        )

        p_encoder, q_encoder = self.train(args, train_dataset, p_encoder, q_encoder)

        p_embedding = []

        for passage in tqdm.tqdm(self.contexts):  # wiki
            passage = self.tokenizer(passage, padding="max_length", truncation=True, return_tensors="pt").to("cuda")
            p_emb = p_encoder(**passage).to("cpu").detach().numpy()
            p_embedding.append(p_emb)

        p_embedding = np.array(p_embedding).squeeze()  # numpy
        return p_embedding, q_encoder


class DprKobertRetrieval(DprRetrieval):
    def __init__(self, args):
        super().__init__(args)
        self.backbone = "monologg/kobert"
        self.tokenizer = KoBertTokenizer.from_pretrained(self.backbone)

    def _load_model(self):
        config = AutoConfig.from_pretrained(self.backbone)
        p_encoder = AutoEncoder(self.backbone, config=config).cuda()
        q_encoder = AutoEncoder(self.backbone, config=config).cuda()
        return p_encoder, q_encoder

    def _get_encoder(self):
        config = AutoConfig.from_pretrained(self.backbone)
        q_encoder = AutoEncoder(self.backbone, config=config).cuda()
        return q_encoder


class DprKorquadBertRetrieval(DprRetrieval):
    def __init__(self, args):
        super().__init__(args)
        self.backbone = "sangrimlee/bert-base-multilingual-cased-korquad"
        self.tokenizer = AutoTokenizer.from_pretrained(self.backbone)

    def _load_model(self):
        config = AutoConfig.from_pretrained(self.backbone)
        p_encoder = AutoEncoder(self.backbone, config=config).cuda()
        q_encoder = AutoEncoder(self.backbone, config=config).cuda()
        return p_encoder, q_encoder

    def _get_encoder(self):
        config = AutoConfig.from_pretrained(self.backbone)
        q_encoder = AutoEncoder(self.backbone, config=config).cuda()
        return q_encoder
