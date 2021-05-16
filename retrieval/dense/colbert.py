import time
import tqdm
import string
import pickle
import os.path as p

import torch
import numpy as np
import torch.nn as nn
from datasets import load_from_disk
from transformers import AdamW, TrainingArguments
from transformers import BertPreTrainedModel, BertModel, BertTokenizerFast, BertConfig


from retrieval.dense import DenseRetrieval


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


class QueryTokenizer:
    def __init__(self):
        self.tok = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")

        self.Q_marker_token, self.Q_marker_token_id = "[Q]", self.tok.convert_tokens_to_ids("[unused0]")
        self.cls_token, self.cls_token_id = self.tok.cls_token, self.tok.cls_token_id
        self.sep_token, self.sep_token_id = self.tok.sep_token, self.tok.sep_token_id
        self.mask_token, self.mask_token_id = self.tok.mask_token, self.tok.mask_token_id

        assert self.Q_marker_token_id == 100 and self.mask_token_id == 103

    def tokenize(self, batch_text, add_special_tokens=False):
        assert type(batch_text) in [list, tuple], type(batch_text)

        tokens = [self.tok.tokenize(x, add_special_tokens=False) for x in batch_text]

        if not add_special_tokens:
            return tokens

        prefix, suffix = [self.cls_token, self.Q_marker_token], [self.sep_token]
        tokens = [prefix + lst + suffix + [self.mask_token] * (self.query_maxlen - (len(lst) + 3)) for lst in tokens]

        return tokens

    def encode(self, batch_text, add_special_tokens=False):
        assert type(batch_text) in [list, tuple], type(batch_text)

        ids = self.tok(batch_text, add_special_tokens=False)["input_ids"]

        if not add_special_tokens:
            return ids

        prefix, suffix = [self.cls_token_id, self.Q_marker_token_id], [self.sep_token_id]
        ids = [prefix + lst + suffix + [self.mask_token_id] * (self.query_maxlen - (len(lst) + 3)) for lst in ids]

        return ids

    def tensorize(self, batch_text, bsize=None):
        assert type(batch_text) in [list, tuple], type(batch_text)

        # add placehold for the [Q] marker
        batch_text = [". " + x for x in batch_text]

        obj = self.tok(
            batch_text, padding="longest", truncation=True, return_tensors="pt", max_length=self.tok.model_max_length
        )

        ids, mask = obj["input_ids"], obj["attention_mask"]

        # postprocess for the [Q] marker and the [MASK] augmentation
        ids[:, 1] = self.Q_marker_token_id
        ids[ids == 0] = self.mask_token_id

        if bsize:
            batches = _split_into_batches(ids, mask, bsize)
            return batches

        return ids, mask


class DocTokenizer:
    def __init__(self, doc_maxlen):
        self.tok = BertTokenizerFast.from_pretrained("bert-base-mulist-cased")
        self.doc_maxlen = doc_maxlen

        self.D_marker_token, self.D_marker_token_id = "[D]", self.tok.convert_tokens_to_ids("[unused1]")
        self.cls_token, self.cls_token_id = self.tok.cls_token, self.tok.cls_token_id
        self.sep_token, self.sep_token_id = self.tok.sep_token, self.tok.sep_token_id

        assert self.D_marker_token_id == 1

    def tokenize(self, batch_text, add_special_tokens=False):
        assert type(batch_text) in [list, tuple], type(batch_text)

        tokens = [self.tok.tokenize(x, add_special_tokens=False) for x in batch_text]

        if not add_special_tokens:
            return tokens

        prefix, suffix = [self.cls_token, self.D_marker_token], [self.sep_token]
        tokens = [prefix + lst + suffix for lst in tokens]

        return tokens

    def encode(self, batch_text, add_special_tokens=False):
        assert type(batch_text) in [list, tuple], type(batch_text)

        ids = self.tok(batch_text, add_special_tokens=False)["input_ids"]

        if not add_special_tokens:
            return ids

        prefix, suffix = [self.cls_token_id, self.D_marker_token_id], [self.sep_token_id]
        ids = [prefix + lst + suffix for lst in ids]

        return ids

    def tensorize(self, batch_text, bsize=None):
        assert type(batch_text) in [list, tuple], type(batch_text)

        # add placehold for the [D] marker
        batch_text = [". " + x for x in batch_text]

        obj = self.tok(
            batch_text, padding="longest", truncation="longest_first", return_tensors="pt", max_length=self.doc_maxlen
        )

        ids, mask = obj["input_ids"], obj["attention_mask"]

        # postprocess for the [D] marker
        ids[:, 1] = self.D_marker_token_id

        if bsize:
            ids, mask, reverse_indices = _sort_by_length(ids, mask, bsize)
            batches = _split_into_batches(ids, mask, bsize)
            return batches, reverse_indices

        return ids, mask


def tensorize_triples(query_tokenizer, doc_tokenizer, queries, positives, negatives, bsize):
    assert len(queries) == len(positives) == len(negatives)
    assert bsize is None or len(queries) % bsize == 0

    N = len(queries)
    Q_ids, Q_mask = query_tokenizer.tensorize(queries)
    D_ids, D_mask = doc_tokenizer.tensorize(positives + negatives)
    D_ids, D_mask = D_ids.view(2, N, -1), D_mask.view(2, N, -1)

    # Compute max among {length of i^th positive, length of i^th negative} for i \in N
    maxlens = D_mask.sum(-1).max(0).values

    # Sort by maxlens
    indices = maxlens.sort().indices
    Q_ids, Q_mask = Q_ids[indices], Q_mask[indices]
    D_ids, D_mask = D_ids[:, indices], D_mask[:, indices]

    (positive_ids, negative_ids), (positive_mask, negative_mask) = D_ids, D_mask

    query_batches = _split_into_batches(Q_ids, Q_mask, bsize)
    positive_batches = _split_into_batches(positive_ids, positive_mask, bsize)
    negative_batches = _split_into_batches(negative_ids, negative_mask, bsize)

    batches = []
    for (q_ids, q_mask), (p_ids, p_mask), (n_ids, n_mask) in zip(query_batches, positive_batches, negative_batches):
        Q = (torch.cat((q_ids, q_ids)), torch.cat((q_mask, q_mask)))
        D = (torch.cat((p_ids, n_ids)), torch.cat((p_mask, n_mask)))
        batches.append((Q, D))

    return batches


def _sort_by_length(ids, mask, bsize):
    if ids.size(0) <= bsize:
        return ids, mask, torch.arange(ids.size(0))

    indices = mask.sum(-1).sort().indices
    reverse_indices = indices.sort().indices

    return ids[indices], mask[indices], reverse_indices


def _split_into_batches(ids, mask, bsize):
    batches = []
    for offset in range(0, ids.size(0), bsize):
        batches.append((ids[offset : offset + bsize], mask[offset : offset + bsize]))

    return batches


class ColBERTEncoder(BertPreTrainedModel):
    def __init__(self, config, mask_punctuation, dim=128, similarity_metric="cosine"):
        super(ColBERTEncoder, self).__init__(config)

        self.similarity_metric = similarity_metric
        self.dim = dim

        self.mask_punctuation = mask_punctuation
        self.skiplist = {}

        if self.mask_punctuation:
            self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")
            self.skiplist = {
                w: True
                for symbol in string.punctuation
                for w in [symbol, self.tokenizer.encode(symbol, add_special_tokens=False)[0]]
            }

        self.bert = BertModel(config)
        self.linear = nn.Linear(config.hidden_size, dim, bias=False)

        self.init_weights()

    def forward(self, Q, D):
        return self.score(self.query(*Q), self.doc(*D))

    def query(self, input_ids, attention_mask, token_type_ids=None):
        input_ids, attention_mask = input_ids.to("gpu"), attention_mask.to("gpu")
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        Q = self.linear(Q)

        return torch.nn.functional.normalize(Q, p=2, dim=2)

    def doc(self, input_ids, attention_mask, token_type_ids=None, keep_dims=True):
        input_ids, attention_mask = input_ids.to("gpu"), attention_mask.to("gpu")
        D = self.bert(input_ids, attention_mask=attention_mask)[0]
        D = self.linear(D)

        mask = torch.tensor(self.mask(input_ids), device="gpu").unsqueeze(2).float()
        D = D * mask

        D = torch.nn.functional.normalize(D, p=2, dim=2)

        if not keep_dims:
            D, mask = D.cpu().to(dtype=torch.float16), mask.cpu().bool().squeeze(-1)
            D = [d[mask[idx]] for idx, d in enumerate(D)]

        return D

    def score(self, Q, D):
        if self.similarity_metric == "cosine":
            return (Q @ D.permute(0, 2, 1)).max(2).values.sum(1)

        assert self.similarity_metric == "l2"
        return (-1.0 * ((Q.unsqueeze(2) - D.unsqueeze(1)) ** 2).sum(-1)).max(-1).values.sum(-1)

    def mask(self, input_ids):
        mask = [[(x not in self.skiplist) and (x != 0) for x in d] for d in input_ids.cpu().tolist()]
        return mask


class ColBert(DenseRetrieval):
    def __init__(self, args):
        super().__init__(args)
        self.query_tokenizer = QueryTokenizer()
        self.doc_tokenizer = DocTokenizer()
        self.dim = 128

    def _load_model(self):
        config = BertConfig.from_pretrained(self.backbone)
        colbert = ColBERTEncoder.from_pretrained(
            config=config, mask_punctuation=string.punctuation, dim=self.dim, similarity_metric="cosine"
        ).cuda()
        return colbert

    def _get_encoder(self):
        config = BertConfig.from_pretrained(self.backbone)
        colbert = ColBERTEncoder.from_pretrained(
            config=config, mask_punctuation=string.punctuation, dim=self.dim, similarity_metric="cosine"
        ).cuda()
        return colbert

    def _load_dataset(self):
        triplet_dataset = load_from_disk(self.args.path.train_data_dir, "triplet_dataset")
        batches = tensorize_triples(
            self.query_tokenizer,
            self.doc_tokenizer,
            triplet_dataset["question"],
            triplet_dataset["context"],
            triplet_dataset["negative"],
            self.retriever.per_device_train_batch_size,
        )

        return batches

    def get_embedding(self):
        if p.isfile(self.embed_path) and p.isfile(self.encoder_path) and not self.args.retriever.retrain:
            with open(self.embed_path, "rb") as f:
                self.p_embedding = pickle.load(f)

            self.encoder = self._get_encoder()
            self.encoder.load_state_dict(torch.load(self.encoder_path))
        else:
            self.p_embedding, self.encoder = self._exec_embedding()

            with open(self.embed_path, "wb") as f:
                pickle.dump(self.p_embedding, f)

            torch.save(self.encoder.state_dict(), self.encoder_path)

    def _train(self, training_args, batches, colbert):
        optimizer = AdamW(filter(lambda p: p.requires_grad, colbert.parameters()), lr=training_args.lr, eps=1e-8)
        optimizer.zero_grad()

        criterion = nn.CrossEntropyLoss()

        colbert.train()
        labels = torch.zeros(30, dtype=torch.long, device="cuda")

        for epoch in range(training_args.num_train_epochs):
            train_loss = 0.0
            start_time = time.time()

            for step, (queries, passages) in enumerate(batches):
                scores = colbert(queries, passages).view(2, -1).permute(1, 0)
                loss = criterion(scores, labels[: scores.size(0)])
                #  loss = loss / training_args.gradient_accumulation_steps

                train_loss += loss.item()

                loss.backward()
                optimizer.step()

                print(f"epoch: {epoch:02} step: {step:02} loss: {loss}", end="\r")

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            print(f"Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s")
            print(f"\tTrain Loss: {train_loss / len(batches):.4f}")

        return colbert

    def _exec_embedding(self):
        colbert = self._load_model()
        batches = self._load_dataset()

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

        colbert = self._train(args, batches, colbert)
        p_embedding = []

        for passage in tqdm.tqdm(self.contexts):  # wiki
            passage = self.doc_tokenizer(
                passage, padding="max_length", truncation=True, max_length=512, return_tensors="pt"
            ).to("cuda")
            p_emb = colbert.query(**passage).to("cpu").detach().numpy()
            p_embedding.append(p_emb)

        p_embedding = np.array(p_embedding).squeeze()  # numpy

        return p_embedding, colbert
