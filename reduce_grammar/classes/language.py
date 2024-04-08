from typing import List, Tuple, Optional, Union, TypeVar, Generic, Dict
from copy import deepcopy
import random
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import math

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

import nltk
from nltk import PCFG as nltk_PCFG

from .config import Config
from .tokenizer import Tokenizer

import logging

logger = logging.getLogger(__name__)


BinaryCorpus = List[Tuple[Tensor, int]]
LMCorpus = List[Tensor]
Corpus = Union[BinaryCorpus, LMCorpus]


class LanguageConfig(Config):
    is_binary: bool = False
    real_output: bool = False
    split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    corpus_size: Optional[int] = None
    corrupt_k: Optional[int] = None
    masked_lm: bool = False
    mask_ratio: float = 0.01
    allow_duplicates: bool = False
    use_unk_pos_tags: bool = False
    resample_unk: bool = False
    file: Optional[str] = None
    store_trees: bool = True
    device: str = "cpu"


C = TypeVar("C", bound=LanguageConfig)


class Language(Generic[C]):
    def __init__(self, config: C, tokenizer: Tokenizer):
        assert sum(config.split_ratio) == 1.0, "train/dev/test split does not add to 1"

        self.config = config
        self.tokenizer = tokenizer
        self.grammar = self.create_grammar()

        if self.config.file is not None:
            logger.info('Loading corpus from file.')
            self.corpus, self.tree_corpus, self.pos_dict = torch.load(self.config.file)
            if self.config.corpus_size is not None and self.config.corpus_size < len(self.corpus):
                # This allows us to subsample the larger corpus directly for smaller corpus sizes
                self.corpus = self.corpus[:self.config.corpus_size]
                self.tree_corpus = {sen: self.tree_corpus[sen] for sen in self.corpus}
                self.pos_dict = {sen: self.pos_dict[sen] for sen in self.corpus}
        else:
            logger.info('Creating corpus from grammar.')
            self.tree_corpus: Dict[str, nltk.Tree] = {}
            self.pos_dict: Dict[str, List[str]] = {}  # maps sentence to pos tags
            self.corpus = self.create_corpus()

        if config.is_binary or config.real_output:
            self.corpus = self.append_corrupt_corpus(self.corpus)

        self.train_corpus, self.dev_corpus, self.test_corpus = self.split()

        self.tokenizer.create_vocab(self.train_corpus, pos_dict=self.pos_dict)

    def __len__(self):
        return len(self.train_corpus) + len(self.dev_corpus) + len(self.test_corpus)

    def save(self, file_name: str) -> None:
        torch.save((self.corpus, self.tree_corpus, self.pos_dict), file_name)

    @property
    def num_symbols(self):
        return len(self.tokenizer.token2idx)

    def create_grammar(self) -> Optional[nltk_PCFG]:
        pass

    def create_corpus(self) -> List[str]:
        raise NotImplementedError

    def make_binary(self, corrupt_grammar, clone=False):
        language = deepcopy(self) if clone else self
        language.config.is_binary = True
        language.config.corrupt_grammar = corrupt_grammar
        language.corpus = language.append_corrupt_corpus(self.corpus)

        language.train_corpus, language.dev_corpus, language.test_corpus = language.split()

        return language

    def make_mlm(self) -> None:
        if not self.config.masked_lm:
            self.config.masked_lm = True
            self.tokenizer.make_masked_post_hoc()
            self.tokenizer.add_cls_post_hoc()

    def append_corrupt_corpus(self, corpus: List[Tensor]) -> List[Tuple[Tensor, int]]:
        corrupt_corpus = []

        for item in corpus:
            corrupt_corpus.append(self._create_corrupt_item(item.clone()))

        # merge corrupt corpus with original one + add labels
        new_corpus = [
            (item, label)
            for items, label in [(corrupt_corpus, 0), (corpus, 1)]
            for item in items
        ]

        return new_corpus

    def _create_corrupt_item(self, item: Tensor, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    def split(self):
        random.shuffle(self.corpus)
        train_ratio, dev_ratio, test_ratio = self.config.split_ratio

        if self.config.allow_duplicates:
            item_distribution = Counter(self.corpus)
            unique_items = list(item_distribution.keys())

            train_split_idx = int(len(item_distribution) * train_ratio)
            dev_split_idx = int(len(item_distribution) * (train_ratio + dev_ratio))
            test_split_idx = int(len(item_distribution) * (train_ratio + dev_ratio + test_ratio))

            train_items = unique_items[:train_split_idx]
            dev_items = unique_items[train_split_idx:dev_split_idx]
            test_items = unique_items[dev_split_idx:test_split_idx]

            # Duplicate each unique item according to the original counts of the item
            train_items = [x for item in train_items for x in [item] * item_distribution[item]]
            dev_items = [x for item in dev_items for x in [item] * item_distribution[item]]
            test_items = [x for item in test_items for x in [item] * item_distribution[item]]
        else:
            train_split_idx = int(len(self.corpus) * train_ratio)
            dev_split_idx = int(len(self.corpus) * (train_ratio + dev_ratio))
            test_split_idx = int(len(self.corpus) * (train_ratio + dev_ratio + test_ratio))

            train_items = self.corpus[:train_split_idx]
            dev_items = self.corpus[train_split_idx:dev_split_idx]
            test_items = self.corpus[dev_split_idx:test_split_idx]

        return train_items, dev_items, test_items

    def batchify(self, *args, corpus: List[str] = None, **kwargs):
        corpus = corpus or self.train_corpus
        if self.config.is_binary:
            corpus, labels = zip(*corpus)
            corpus = list(corpus)

        tokenized_corpus = self.tokenize_corpus(corpus)

        if self.config.is_binary:
            return self.batchify_binary(tokenized_corpus, labels, *args, **kwargs)
        elif self.config.masked_lm:
            return self.batchify_masked(tokenized_corpus, *args, **kwargs)
        else:
            return self.batchify_generative(tokenized_corpus, *args, **kwargs)

    def tokenize_corpus(self, corpus: List[str]) -> List[Tensor]:
        if self.config.resample_unk:
            return [
                self.tokenizer.tokenize(
                    sen, 
                    pos_tags=self.tree_to_pos(self.tree_corpus[sen], merge=False), 
                    grammar=self.grammar
                )
                for sen in corpus
            ]
        else:
            return [
                self.tokenizer.tokenize(sen, pos_tags=self.pos_dict.get(sen))
                for sen in corpus
            ]

    def batchify_binary(self, corpus: List[Tensor], labels, batch_size=None):
        batch_size = batch_size or len(corpus)

        corpus_lengths = torch.tensor([x.shape[0] for x in corpus])

        padded_corpus = pad_sequence(corpus, batch_first=True, padding_value=self.tokenizer.pad_idx)

        for idx in range(0, len(corpus), batch_size):
            input_ids = padded_corpus[idx: idx + batch_size]
            lengths = corpus_lengths[idx: idx + batch_size]

            targets = torch.tensor(labels[idx: idx + batch_size])

            yield input_ids.to(self.config.device), targets.to(self.config.device).float(), lengths, None

    def batchify_masked(self, corpus: List[Tensor], batch_size=None):
        batch_size = batch_size or len(corpus)

        corpus_lengths = torch.tensor([len(x) for x in corpus])

        padded_corpus = pad_sequence(corpus, batch_first=True, padding_value=self.tokenizer.pad_idx)

        for idx in range(0, len(corpus), batch_size):
            input_ids = padded_corpus[idx: idx + batch_size]
            lengths = corpus_lengths[idx: idx + batch_size]

            masks = [
                random.sample(
                    range(int(self.tokenizer.config.add_cls), length),
                    k=math.ceil(self.config.mask_ratio * length)
                )
                for length in lengths
            ]
            targets = []

            for jdx, mask in enumerate(masks):
                targets.extend(input_ids[jdx, mask].tolist())
                input_ids[jdx, mask] = self.tokenizer.mask_idx

            targets = torch.tensor(targets)

            yield input_ids.to(self.config.device), targets.to(self.config.device), lengths, masks

    def batchify_generative(self, corpus: List[Tensor], batch_size=None):
        batch_size = batch_size or len(corpus)

        corpus_lengths = torch.tensor([len(x) for x in corpus])

        padded_corpus = pad_sequence(corpus, batch_first=True, padding_value=self.tokenizer.pad_idx)

        for idx in range(0, len(corpus), batch_size):
            items = padded_corpus[idx: idx + batch_size]
            lengths = corpus_lengths[idx: idx + batch_size] - 1

            input_ids = items[:, :-1]
            targets = items[:, 1:]
            flat_targets = self._unpad_sequence(targets, lengths)

            yield input_ids.to(self.config.device), flat_targets.to(self.config.device), lengths, None

    def gen_baselines(*args, **kwargs):
        raise NotImplementedError

    def plot_sen_len_distribution(self, title=None):
        item_lens = [len(x) for x in self.corpus]
        sns.histplot(item_lens, color="orange", bins=range(min(item_lens), max(item_lens)))
        if self.config.allow_duplicates:
            unique_items = set(self.corpus)
            unique_lens = [len(x) for x in [x.split(self.tokenizer.config.sep_token) for x in unique_items]]
            sns.histplot(unique_lens, color="yellow", alpha=0.5, bins=range(min(item_lens), max(item_lens)))

        plt.xlabel("Sentence length")
        if title:
            plt.title(title)

        plt.show()
    
    def tree_to_pos(self, tree, merge=True):
        if merge:
            return [
                prod.lhs().symbol().split('_')[0] for prod in tree.productions()
                if isinstance(prod.rhs()[0], str)
            ]
        else:
            return [
                prod.lhs().symbol() for prod in tree.productions()
                if isinstance(prod.rhs()[0], str)
            ]
    

    def _unpad_sequence(self, tensor, lengths, concat=True):
        """Casts a padded tensor to a concated tensor that omits the pad positions."""
        concat_fn = torch.stack if tensor.ndim > 2 else torch.tensor
        tensor_list = [value for row, idx in zip(tensor, lengths) for value in row[:idx]]

        return concat_fn(tensor_list).to(self.config.device)