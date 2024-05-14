import numpy as np
from typing import List, Dict
import json
import os

from transformers import AutoModelForMaskedLM, AutoModelForCausalLM
from tokenizer import create_tf_tokenizer_from_vocab
import logging

logger = logging.getLogger(__name__)

def load_model(checkpoint, device):
    model = AutoModelForMaskedLM.from_pretrained(checkpoint).to(device)

    with open(f'{checkpoint}/added_tokens.json') as f:
        vocab = json.load(f)
    tokenizer = create_tf_tokenizer_from_vocab(vocab)

    return model, tokenizer


def load_model_tokenizer(args):
    if args.model == 'gpt2':
        automodel = AutoModelForCausalLM
    elif args.model == 'babyberta':
        automodel = AutoModelForMaskedLM
    else:
        raise ValueError(f'Model {args.model} not supported.')
    
    model = automodel.from_pretrained(args.model_file)

    with open(f'{args.model_file}added_tokens.json') as f:
        vocab = json.load(f)

    tokenizer = create_tf_tokenizer_from_vocab(vocab)

    return model, tokenizer


def set_experiment_config(args):
    # Check if model directory exists and find the correct checkpoint
    if not os.path.exists(args.model_path):
        raise ValueError(f'Model directory {args.model_path} does not exist.')
    
    highest_config = 0
    print(args.model_path)
    for dir_name in os.listdir(args.model_path):
        if dir_name.split('-')[0] == 'checkpoint':
            config = int(dir_name.split('-')[1])
            if config > highest_config:
                highest_config = config

    args.model_file = f'{args.model_path}/checkpoint-{highest_config}/'

    return args
