from typing import Optional, Tuple

import torch
import numpy as np

from fairseq.models.roberta import RobertaHubInterface, RobertaModel
from pororo.tasks.utils.base import PororoBiencoderBase, PororoFactoryBase
from pororo.tasks.utils.download_utils import download_or_load
from pororo.tasks.utils.tokenizer import CustomTokenizer


class PororoMrcFactory(PororoFactoryBase):
    def __init__(self, task: str, lang: str, model: Optional[str]):
        super().__init__(task, lang, model)

    @staticmethod
    def get_available_langs():
        return ["ko"]

    @staticmethod
    def get_available_models():
        return {"ko": ["brainbert.base.ko.korquad"]}

    def load(self, device: str):

        if "brainbert" in self.config.n_model:
            try:
                import mecab
            except ModuleNotFoundError as error:
                raise error.__class__("Please install python-mecab-ko with: `pip install python-mecab-ko`")
            # from pororo.models.brainbert import BrainRobertaModel
            from pororo.utils import postprocess_span

            model = My_BrainRobertaModel.load_model(f"bert/{self.config.n_model}", self.config.lang).eval().to(device)

            tagger = mecab.MeCab()

            return PororoBertMrc(model, tagger, postprocess_span, self.config)


class My_BrainRobertaModel(RobertaModel):
    @classmethod
    def load_model(cls, model_name: str, lang: str, **kwargs):
        from fairseq import hub_utils

        ckpt_dir = download_or_load(model_name, lang)
        tok_path = download_or_load(f"tokenizers/bpe32k.{lang}.zip", lang)

        x = hub_utils.from_pretrained(ckpt_dir, "model.pt", ckpt_dir, load_checkpoint_heads=True, **kwargs)
        return BrainRobertaHubInterface(x["args"], x["task"], x["models"][0], tok_path)


class BrainRobertaHubInterface(RobertaHubInterface):
    def __init__(self, args, task, model, tok_path):
        super().__init__(args, task, model)
        self.bpe = CustomTokenizer.from_file(
            vocab_filename=f"{tok_path}/vocab.json", merges_filename=f"{tok_path}/merges.txt"
        )

    def tokenize(self, sentence: str, add_special_tokens: bool = False):
        result = " ".join(self.bpe.encode(sentence).tokens)
        if add_special_tokens:
            result = f"<s> {result} </s>"
        return result

    def encode(
        self, sentence: str, *addl_sentences, add_special_tokens: bool = True, no_separator: bool = False
    ) -> torch.LongTensor:

        bpe_sentence = self.tokenize(sentence, add_special_tokens=add_special_tokens)

        for s in addl_sentences:
            bpe_sentence += " </s>" if not no_separator and add_special_tokens else ""
            bpe_sentence += " " + self.tokenize(s, add_special_tokens=False) + " </s>" if add_special_tokens else ""
        tokens = self.task.source_dictionary.encode_line(bpe_sentence, append_eos=False, add_if_not_exist=False)
        return tokens.long()

    def decode(self, tokens: torch.LongTensor, skip_special_tokens: bool = True, remove_bpe: bool = True) -> str:
        assert tokens.dim() == 1
        tokens = tokens.numpy()

        if tokens[0] == self.task.source_dictionary.bos() and skip_special_tokens:
            tokens = tokens[1:]  # remove <s>

        eos_mask = tokens == self.task.source_dictionary.eos()
        doc_mask = eos_mask[1:] & eos_mask[:-1]
        sentences = np.split(tokens, doc_mask.nonzero()[0] + 1)

        if skip_special_tokens:
            sentences = [np.array([c for c in s if c != self.task.source_dictionary.eos()]) for s in sentences]

        sentences = [" ".join([self.task.source_dictionary.symbols[c] for c in s]) for s in sentences]

        if remove_bpe:
            sentences = [s.replace(" ", "").replace("▁", " ").strip() for s in sentences]
        if len(sentences) == 1:
            return sentences[0]
        return sentences

    @torch.no_grad()
    def predict_span(
        self, question: str, context: str, add_special_tokens: bool = True, no_separator: bool = False
    ) -> Tuple:

        max_length = self.task.max_positions()
        tokens = self.encode(question, context, add_special_tokens=add_special_tokens, no_separator=no_separator)[
            :max_length
        ]
        with torch.no_grad():
            logits = self.predict("span_prediction_head", tokens, return_logits=True).squeeze()  # T x 2

            results = []
            top_n = 1  # n*n개 return

            starts = logits[:, 0].argsort(descending=True)[:top_n].tolist()

            for start in starts:
                ends = logits[:, 1].argsort(descending=True).tolist()
                masked_ends = [end for end in ends if end >= start]
                ends = (masked_ends + ends)[:top_n]  # masked_ends가 비어있을 경우를 대비
                for end in ends:
                    answer_tokens = tokens[start : end + 1]
                    answer = ""
                    if len(answer_tokens) >= 1:
                        decoded = self.decode(answer_tokens)
                        if isinstance(decoded, str):
                            answer = decoded

                    score = (logits[:, 0][start] + logits[:, 1][end]).item()
                    results.append((answer, (start, end + 1), score))

        return results


class PororoBertMrc(PororoBiencoderBase):
    def __init__(self, model, tagger, callback, config):
        super().__init__(config)
        self._model = model
        self._tagger = tagger
        self._callback = callback

    def predict(self, query: str, context: str, **kwargs) -> Tuple[str, Tuple[int, int]]:
        postprocess = kwargs.get("postprocess", True)

        ###
        pair_results = self._model.predict_span(query, context)
        returns = []

        for pair_result in pair_results:
            span = self._callback(self._tagger, pair_result[0]) if postprocess else pair_result[0]
            returns.append((span, pair_result[1], pair_result[2]))

        return returns
