import numpy as np
from typing import List, Dict
import json
import torch
import unicodedata

from transformers import AutoModelForMaskedLM, AutoModelForCausalLM
from tokenizer import create_tf_tokenizer_from_vocab
import logging
import pickle

logger = logging.getLogger(__name__)

def load_model(config):
    checkpoint = config.model_path
    if config.model == 'gpt2':
        model = AutoModelForCausalLM.from_pretrained(checkpoint).to(config.device)
    elif config.model == 'deberta':
        model = AutoModelForMaskedLM.from_pretrained(checkpoint).to(config.device)
    else:
        raise ValueError(f'Unknown model: {config.model}')

    with open(f'{checkpoint}/added_tokens.json') as f:
        vocab = json.load(f)
    tokenizer = create_tf_tokenizer_from_vocab(vocab)

    return model, tokenizer

def get_all_subword_id(mapping, idx):
    current_id = mapping[idx]
    id_for_all_subwords = [tmp_id for tmp_id, v in enumerate(mapping) if v == current_id]

    return id_for_all_subwords

def find_root(parse):
    # root node's head also == 0, so have to be removed
    for token in parse[1:]:
        if token.head == 0:
            return token.id
    return False


def _run_strip_accents(text):
    """Strips accents from a piece of text."""
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            continue
        output.append(char)
    return "".join(output)


def match_tokenized_to_untokenized(subwords, sentence):
    token_subwords = np.zeros(len(sentence))
    #sentence = [_run_strip_accents(x) for x in sentence]
    token_ids, subwords_str, current_token, current_token_normalized = [-1] * len(subwords), "", 0, None
    for i, subword in enumerate(subwords):
        if subword in ["[CLS]", "[SEP]"]: continue

        while current_token_normalized is None:
            current_token_normalized = sentence[current_token]

        if subword.startswith("[UNK]"):
            unk_length = int(subword[6:])
            subwords[i] = subword[:5]
            subwords_str += current_token_normalized[len(subwords_str):len(subwords_str) + unk_length]
        else:
            subwords_str += subword[2:] if subword.startswith("##") else subword
        if not current_token_normalized.startswith(subwords_str):
            logger.warning('Tokenization mismatch: %s vs %s', current_token_normalized, subwords_str)
            return False

        token_ids[i] = current_token
        token_subwords[current_token] += 1
        if current_token_normalized == subwords_str:
            subwords_str = ""
            current_token += 1
            current_token_normalized = None

    assert current_token_normalized is None
    while current_token < len(sentence):
        assert not sentence[current_token]
        current_token += 1
    assert current_token == len(sentence)

    return token_ids


def write_to_file(out, config):
    """
    Write the impact matrices per layer to a seperate file
    """
    for k, one_layer_out in enumerate(out):
        k_output = config.output_dir / f'{config.model}_{config.metric}_{str(k)}.pkl'
        with open(k_output, 'wb') as fout:
            pickle.dump(out[k], fout)
            fout.close()
