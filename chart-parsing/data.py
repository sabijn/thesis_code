from transformers import PreTrainedModel
from tqdm import *
from collections import defaultdict
import torch
import nltk
import random
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Tuple
import pickle
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def tree_to_pos(tree, tokenizer, skip_unk_tokens=False):
    pos_tags = [
        prod.lhs().symbol().split("_")[0]
        for prod in tree.productions()
        if isinstance(prod.rhs()[0], str)
    ]
    assert len(pos_tags) == len(tree.leaves())
    if skip_unk_tokens:
        no_unk_pos = []
        for pos, w in zip(pos_tags, tree.leaves()):
            if w in tokenizer.vocab:
                no_unk_pos.append(pos)
        return no_unk_pos
    else:
        return pos_tags

def tree_to_spanlabels(tree, device, merge_pos=False, merge_at=False, skip_labels=None):
    """ Returns the spans of all subtrees and their labels """
    sen_len = len(tree.leaves())
    treeposition_to_span = defaultdict(list)
    treeposition_to_span[()] = [(0, sen_len-1)]

    for i in range(sen_len):
        for j in range(i+2, sen_len):
            treeposition = tree.treeposition_spanning_leaves(i, j)
            span = (i, j - 1) # minus 1 because in self-attention span ends are inclusive
            treeposition_to_span[treeposition].append(span)  

    for treeposition, span in treeposition_to_span.items():
        treeposition_to_span[treeposition] = max(span, key=lambda x: x[1] - x[0])

    nonterminal_treeposition = [
        treeposition
        for treeposition in tree.treepositions()
        if isinstance(tree[treeposition], nltk.Tree)
    ]

    span_ids = torch.zeros(len(treeposition_to_span), 2, device=device).long()
    labels = []

    for idx, (treeposition, span) in enumerate(treeposition_to_span.items()):
        subtree = tree[treeposition]
        start, end = treeposition_to_span[treeposition]

        span_ids[idx, 0] = start
        span_ids[idx, 1] = end

        label = subtree.label()
        if merge_pos:
            label = label.split("_")[0]
        if merge_at:
            label = label.replace("AT", "")
    
        labels.append(label)
        
    if skip_labels is not None:
        label_mask = [label not in skip_labels for label in labels]
        labels = [label for label, mask in zip(labels, label_mask) if mask]
        span_ids = span_ids[label_mask]

    return span_ids, labels
    

def extract_span_labels(tree_corpus, 
                        device, 
                        merge_pos=False, 
                        merge_at=False, 
                        skip_labels=None) -> Tuple[List[torch.Tensor], List[torch.Tensor], dict]:
    all_span_ids = []
    all_labels = []
    
    for tree in tqdm(tree_corpus):
        span_ids, labels = tree_to_spanlabels(
            tree, device, merge_pos=merge_pos, merge_at=merge_at, skip_labels=skip_labels
        )
        all_span_ids.append(span_ids)
        all_labels.append(labels)
    
    unique_labels = set([label for labels in all_labels for label in labels])
    label_vocab = {
        label: idx
        for idx, label in enumerate(unique_labels)
    }
    all_tokenized_labels = []

    for labels in all_labels:
        tokenized_labels = torch.zeros(len(labels), device=device).long()
        for idx, label in enumerate(labels):
            tokenized_labels[idx] = label_vocab[label]
        all_tokenized_labels.append(tokenized_labels)
    
    return all_span_ids, all_tokenized_labels, label_vocab


def create_states(
    tokenizer, 
    tree_corpus, 
    model, 
    config,
    concat=True, 
    skip_cls=False, 
    num_items=None,
    verbose=False,
    all_layers=False,
    skip_unk_tokens=False,
):
    if isinstance(model, PreTrainedModel):
        all_sens = [torch.tensor(tokenizer.convert_tokens_to_ids(tree.leaves())) for tree in tree_corpus]
        pad_idx = tokenizer.pad_token_id
        num_parameters = model.num_parameters()
    else:
        all_sens = [tokenizer.tokenize(tree.leaves(), pos_tags=tree_to_pos(tree, tokenizer)) for tree in tree_corpus]
        pad_idx = tokenizer.pad_idx
        num_parameters = model.num_parameters

    if num_items is not None:
        all_sens = random.sample(all_sens, num_items)
    lengths = [len(sen) for sen in all_sens]
    sen_tensor = pad_sequence(all_sens, padding_value=pad_idx, batch_first=True).to(config.device)

    batch_size = int(1e9 / num_parameters)
    states = defaultdict(list) if all_layers else []
    iterator = range(0, len(all_sens), batch_size)
    if verbose:
        iterator = tqdm(iterator)

    for idx in iterator:
        batch = sen_tensor[idx: idx + batch_size]

        with torch.no_grad():
            all_hidden = model(batch, output_hidden_states=True).hidden_states

        if all_layers:
            for layer_idx, layer_hidden in enumerate(all_hidden):
                for hidden, sen, length in zip(layer_hidden, batch, lengths[idx: idx + batch_size]):
                    unk_mask = sen[:length] != tokenizer.unk_token_id
                    states[layer_idx].append(hidden[:length][unk_mask])
        else:
            states.extend([
                hidden[int(skip_cls):length]
                for hidden, length in zip(all_hidden[-1], lengths[idx: idx + batch_size])
            ])
    
    if all_layers:
        pickle.dump(states, open(config.created_states_path / Path('states_all_layers.pkl'), 'wb'))
    else:
        pickle.dump(states, open(config.created_states_path / Path('states.pkl'), 'wb'))

    if concat:
        if all_layers:
            for layer_idx, layer_states in states.items():
                states[layer_idx] = torch.concat(layer_states)
            return states
        else:
            return torch.concat(states)
    else:
        return states
    
def create_data_splits(data):
    corpus_size = len(data)
    train_split, dev_split, test_split = int(0.8 * corpus_size), int(0.9 * corpus_size), corpus_size

    train_data = data[:train_split]
    dev_data = data[train_split:dev_split]
    test_data = data[dev_split:test_split]

    return train_data, dev_data, test_data