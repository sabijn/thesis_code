from .config import Config
from .model import LanguageClassifier
from .language import Language

from typing import List, Dict, Any, Union, Optional
from copy import deepcopy
import math
import random
from tqdm import tqdm
from collections import defaultdict
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import (
    pad_sequence,
)

import matplotlib.pyplot as plt

class ExperimentConfig(Config):
    lr: float = 1e-2
    batch_size: int = 48
    epochs: int = 10
    verbose: bool = False
    early_stopping: Optional[int] = None
    continue_after_optimum: int = 0
    eval_every: int = 100
    warmup_duration: int = 0
    eval_dev_pos_performance: bool = False
    eval_test_pos_performance: bool = False

    
class ExitException(Exception):
    pass


class Experiment:
    def __init__(self, model: LanguageClassifier, config: ExperimentConfig) -> None:
        self.model = model
        self.config = config
        self.best_model = None

    def save(self, filename):
        torch.save(self, filename)
        print("Saved experiment to", filename)

    def train(self, languages: Union[Language, List[Language]]):
        if not isinstance(languages, list):
            languages = [languages]

        performances: Dict[int, Dict[str, Any]] = {}

        for lang_idx, language in enumerate(languages):
            performances[lang_idx] = self._train_language(language)

            self.model = self.best_model

        return performances

    def _train_language(self, language: Language) -> Dict[str, Any]:
        optimizer = optim.AdamW(self.model.parameters(), lr=self.config.lr)
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer, 
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.config.warmup_duration,
        )
        if language.config.real_output:
            loss_function = nn.MSELoss()
        elif language.config.is_binary:
            loss_function = nn.BCEWithLogitsLoss()
        else:
            loss_function = nn.CrossEntropyLoss()

        performance_scores = {
            'train': [],
            'dev': [],
            'test': None,
        }
        
        if self.config.eval_dev_pos_performance:
            performance_scores['dev_pos'] = []

        best_dev_acc = -math.inf
        batches_seen = 0
        batches_seen_at_best_dev_acc = 0
        batch_optimum = 0

        if self.config.verbose:
            iterator = tqdm(range(self.config.epochs))
        else:
            iterator = range(self.config.epochs)

        try:
            for _epoch in iterator:
                random.shuffle(language.train_corpus)

                for batch in language.batchify(
                    corpus=language.train_corpus, batch_size=self.config.batch_size,
                ):
                    self._update_model(batch, optimizer, loss_function)

                    batches_seen += 1

                    if batches_seen % self.config.eval_every == 0:
                        best_dev_acc, batches_seen_at_best_dev_acc = self._eval(
                            batches_seen,
                            language,
                            performance_scores,
                            best_dev_acc,
                            batches_seen_at_best_dev_acc,
                        )

                        time_since_improvement = batches_seen - batches_seen_at_best_dev_acc
                        if (
                            self.config.early_stopping is not None
                            and time_since_improvement > self.config.early_stopping
                        ):
                            # Stop if no increases have been registered for past X epochs
                            if self.config.verbose:
                                print("Stopping early...")
                            raise ExitException
                        elif best_dev_acc == 1.0 and batch_optimum == 0:
                            if self.config.verbose:
                                print(f"Optimum reached at iteration {batches_seen}")
                            batch_optimum = batches_seen
                        
                        if best_dev_acc == 1.0 and ((batch_optimum + self.config.continue_after_optimum) == batches_seen):
                            raise ExitException
                        
                        scheduler.step()
                            
        except (KeyboardInterrupt, ExitException) as e:
            pass

        if len(language.test_corpus) > 0:
            performance_scores['test'] = self.eval_corpus(language, language.test_corpus, model=self.best_model)

            if self.config.eval_test_pos_performance and not language.config.is_binary:
                performance_scores['test_pos'] = self.pos_performance(language, language.test_corpus, self.model)

        return performance_scores

    def _update_model(self, batch, optimizer, loss_function) -> None:
        input_ids, targets, lengths, mask_ids = batch

        self.model.zero_grad()

        predictions = self.model(input_ids=input_ids, input_lengths=lengths, mask_ids=mask_ids)

        loss = loss_function(predictions, targets)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)

        optimizer.step()

    def _eval(
        self,
        batches_seen,
        language,
        performance_scores,
        best_dev_acc,
        batches_seen_at_best_dev_acc,
    ):
        train_acc = self.eval_corpus(language, language.train_corpus)
        performance_scores['train'].append(train_acc)

        dev_acc = self.eval_corpus(language, language.dev_corpus)
        performance_scores['dev'].append(dev_acc)

        if self.config.eval_dev_pos_performance:
            dev_pos_probs = self.pos_performance(language, language.dev_corpus, self.model)
            performance_scores['dev_pos'].append(dev_pos_probs)
        
        if dev_acc > best_dev_acc:
            self.best_model = deepcopy(self.model)
            best_dev_acc = dev_acc
            batches_seen_at_best_dev_acc = batches_seen
            if self.config.verbose:
                print(
                    f"New best at iteration {batches_seen}, dev acc: {dev_acc:.4f}, "
                    f"train acc: {train_acc:.4f}"
                )

        return best_dev_acc, batches_seen_at_best_dev_acc

    def eval_corpus(self, language, corpus, model=None):
        model = model or self.model

        model.eval()

        correct = 0

        batch_size = int(1e9 / model.num_parameters)
        
        for input_ids, targets, lengths, mask_ids in language.batchify(
            corpus=corpus, batch_size=batch_size
        ):
            with torch.no_grad():
                raw_predictions = model(input_ids=input_ids, input_lengths=lengths, mask_ids=mask_ids)

            if language.config.real_output:
                loss_function = nn.MSELoss()
                correct += -loss_function(raw_predictions, targets)
            elif self.model.is_binary:
                predictions = (raw_predictions > 0).to(int)
                correct += int(sum(predictions == targets))
            elif mask_ids is not None:
                ce_loss = F.cross_entropy(raw_predictions, targets)
                perplexity = -ce_loss.exp()

                correct += perplexity * len(mask_ids)
            else:
                split_predictions = torch.split(raw_predictions, tuple(lengths))
                targets = torch.split(targets, tuple(lengths))
                
                for raw_prediction, target, length in zip(split_predictions, targets, lengths):
                    ## [..] Old evaluation setup for dyck/palindromes in previous notebook
                    ce_loss = F.cross_entropy(raw_prediction, target)
                    perplexity = -ce_loss.exp()

                    correct += perplexity

        model.train()

        performance = (correct / len(corpus))

        if isinstance(performance, torch.Tensor):
            performance = performance.item()

        return performance

    def eval_corpora(
        self, 
        languages: List[Language], 
        model: Optional[LanguageClassifier] = None, 
        lang_names: Optional[List[str]] = None, 
        indomain_langs: Optional[List[int]] = None,
        xlabel: str = "Language",
        plot: bool = True,
    ):
        model = model or self.model
        
        accuracies = {}
        
        for lang_idx, language in enumerate(languages):
            accuracy = self.eval_corpus(language, language.corpus, model=model)
            
            lang_name = lang_names[lang_idx] if lang_names else repr(language)
            accuracies[lang_name] = accuracy
            
            if isinstance(accuracy, torch.Tensor):
                accuracy = accuracy.item()
            if self.config.verbose:
                print(f"{lang_name}\t{accuracy:.4f}")

        if plot:
            plot_eval_corpora(accuracies, indomain_langs or [], xlabel)
            
        return accuracies  
    
    
    def pos_prob_mass(self, preds, tree, tokenizer, model, pos_to_words):
        pos_seq = [x for x in tree_to_pos(tree, merge=False)]

        prob_masses = []

        for i in range(int(tokenizer.config.add_cls), len(pos_seq)):            
            word_ids = [
                tokenizer.token2idx[word] 
                for word in pos_to_words[pos_seq[i]] 
                if word in tokenizer.token2idx
            ]
            sum_prob = torch.sum(self.get_word_prob(model, tokenizer, i, word_ids, preds)).item()

            main_pos = pos_seq[i].split('_')[0]
            word_ids = [
                tokenizer.token2idx[word] 
                for word in pos_to_words[main_pos] 
                if word in tokenizer.token2idx
            ]
            sum_prob_merged = torch.sum(self.get_word_prob(model, tokenizer, i, word_ids, preds)).item()

            prob_masses.append((pos_seq[i], main_pos, sum_prob, sum_prob_merged))

        return prob_masses
    
    
    def get_word_prob(self, model, tokenizer, idx, word_ids: List[int], preds):
        if isinstance(model.encoder, nn.LSTM):
            # -1 for auto-regressive nature
            return preds[idx-1, word_ids]
        elif tokenizer.config.add_cls:
            # +1 to account for [CLS]
            return preds[idx+1, word_ids]
        else:
            return preds[idx, word_ids]


    def pos_performance(self, language, corpus: List[str], model):
        pos_to_words = defaultdict(set)
        for prod in language.grammar.productions():
            if isinstance(prod.rhs()[0], str):
                word = prod.rhs()[0]
                pos = prod.lhs().symbol()
                main_pos = pos.split('_')[0]

                pos_to_words[pos].add(word)
                pos_to_words[main_pos].add(word)

        pos_probs = defaultdict(list)
        main_pos_probs = defaultdict(list)
                
        sen_idx = 0

        batch_size = int(1e9 / model.num_parameters)
        tokenized_corpus = language.tokenize_corpus(corpus)
        padded_corpus = pad_sequence(tokenized_corpus, batch_first=True, padding_value=language.tokenizer.pad_idx)
        
        for batch_input in padded_corpus.split(batch_size):
            with torch.no_grad():
                batch_preds = model(batch_input, pseudo_ll=True).softmax(-1)

            for preds in batch_preds:
                tree = language.tree_corpus[corpus[sen_idx]]
            
                prob_mass = self.pos_prob_mass(preds, tree, language.tokenizer, model, pos_to_words)

                for pos, main_pos, pos_prob, main_pos_prob in prob_mass:
                    pos_probs[pos].append(pos_prob)
                    main_pos_probs[main_pos].append(main_pos_prob)
                    
                sen_idx += 1
            
        avg_main_pos_probs = {pos: np.mean(probs) for pos, probs in main_pos_probs.items()}

        return avg_main_pos_probs      


def plot_eval_corpora(accuracies: Dict[str, float], indomain_langs: List[int], xlabel: str):
    id_color = "#40B0A6"
    ood_color = "#E1BE6A"

    colors = [id_color if idx in indomain_langs else ood_color for idx in range(len(accuracies))]

    plt.bar(accuracies.keys(), accuracies.values(), color=colors)
    plt.ylabel("Accuracy")
    plt.xlabel(xlabel)
    plt.ylim(-0.02, 1.01)

    legend_elements = [
        plt.Line2D([0], [0], color=id_color, lw=10, label="Trained on (ID)"),
        plt.Line2D([0], [0], lw=10, color=ood_color, label="OOD"),
    ]
    legend = plt.legend(
        handles=legend_elements, bbox_to_anchor=(1.03, 1), loc="upper left"
    )
    frame = legend.get_frame()
    frame.set_facecolor("w")
    frame.set_edgecolor("black")

    plt.show()


# def train(model, language, config):
#     experiment = Experiment(
#         model,
#         config,
#     )

#     return experiment, experiment.train(language)


def tree_to_pos(tree, merge=True):
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
