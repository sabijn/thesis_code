from typing import List, Dict, Union, Optional
from nltk import PCFG as nltk_PCFG
from nltk.grammar import Nonterminal
from collections import Counter
import random

import torch
from torch import Tensor

from .config import Config

class TokenizerConfig(Config):
    cls_token = "[CLS]"
    mask_token = "<mask>"
    pad_token = "<pad>"
    unk_token = "<unk>"
    unk_threshold: Optional[int] = None
    add_cls: bool = False
    masked_lm: bool = False
    sep_token: str = " "
    resample_unk: bool = False
    device: str = "cpu"


class Vocab(dict):
    def __init__(self, unk_idx: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.unk_idx = unk_idx

    def __missing__(self, key):
        return self.unk_idx


class Tokenizer:
    def __init__(
        self, config: TokenizerConfig
    ):
        self.config = config

        self.idx2token: List[str] = [self.config.unk_token, self.config.pad_token]
        self.token2idx: Dict[str, int] = Vocab(0)

    @property
    def cls_idx(self):
        return self.token2idx.get(self.config.cls_token)

    @property
    def mask_idx(self):
        return self.token2idx.get(self.config.mask_token)

    @property
    def unk_idx(self):
        return self.token2idx.get(self.config.unk_token)

    @property
    def pad_idx(self):
        return self.token2idx.get(self.config.pad_token)

    def create_vocab(self, str_corpus: List[str], pos_dict=None):
        if self.config.masked_lm:
            self.idx2token.append(self.config.mask_token)

        if self.config.add_cls:
            self.idx2token.append(self.config.cls_token)

        if self.config.unk_threshold is not None:
            distribution = Counter(
                w for s in str_corpus for w in s.split(self.config.sep_token)
            )
            self.idx2token.extend(
                [
                    token
                    for token, counts in distribution.items()
                    if counts > self.config.unk_threshold
                ]
            )
        else:
            unique_tokens = set(
                w for s in str_corpus for w in s.split(self.config.sep_token)
            )
            self.idx2token.extend(list(unique_tokens))

        if pos_dict is not None:
            unique_pos = set(
                pos_tag for pos_tags in pos_dict.values() for pos_tag in pos_tags
            )
            self.idx2token.extend(list(unique_pos))

        self.token2idx.update({x: idx for idx, x in enumerate(self.idx2token)})

    def tokenize(
        self, 
        item: Union[str, List[str]], 
        pos_tags: Optional[List[str]] = None,
        grammar: Optional[nltk_PCFG] = None,
    ) -> Tensor:
        if isinstance(item, str):
            item = item.split(self.config.sep_token)

        token_ids = []
        if self.config.add_cls and item[0] != self.config.cls_token:
            token_ids.append(self.cls_idx)

        if pos_tags is None:
            token_ids.extend([self.token2idx[w] for w in item])
        elif grammar is not None:
            assert len(item) == len(pos_tags)
            for w, pos_tag in zip(item, pos_tags):
                if w in self.token2idx:
                    token_ids.append(w)
                else:
                    lhs = Nonterminal(pos_tag)
                    rules = grammar._lhs_index[lhs]
                    probs = grammar._lhs_prob_index[lhs]
                    new_idx = None
                    while new_idx is not None:
                        # Sample until we got a new token that is in the vocab
                        new_token = random.choices(rules, weights=probs, k=1)[0].rhs()[0]
                        new_idx = self.token2idx.get(new_token)
                    token_ids.append(new_idx)
        else:
            assert len(item) == len(pos_tags)
            token_ids.extend(
                [
                    self.token2idx.get(w, self.token2idx[pos_tag])
                    for w, pos_tag in zip(item, pos_tags)
                ]
            )

        return torch.tensor(token_ids, device=self.config['device'])

    def translate(self, item: Tensor, omit_cls: bool = False) -> str:
        sen_list = [self.idx2token[idx] for idx in item.tolist()]
        if omit_cls:
            sen_list = sen_list[1:]

        return self.config.sep_token.join(sen_list)

    def make_masked_post_hoc(self) -> None:
        self.config.masked_lm = True
        if self.config.mask_token not in self.token2idx:
            self.token2idx[self.config.mask_token] = len(self.token2idx)
            self.idx2token.append(self.config.mask_token)

    def add_cls_post_hoc(self) -> None:
        self.config.add_cls = True

        if self.config.cls_token not in self.token2idx:
            self.token2idx[self.config.cls_token] = len(self.token2idx)
            self.idx2token.append(self.config.cls_token)