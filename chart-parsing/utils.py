import numpy as np
from typing import List, Dict
import json
import torch

from transformers import AutoModelForMaskedLM
from tokenizer import create_tf_tokenizer_from_vocab
import logging

logger = logging.getLogger(__name__)

def load_model(checkpoint, device):
    model = AutoModelForMaskedLM.from_pretrained(checkpoint).to(device)

    with open(f'{checkpoint}/added_tokens.json') as f:
        vocab = json.load(f)
    tokenizer = create_tf_tokenizer_from_vocab(vocab)

    return model, tokenizer
