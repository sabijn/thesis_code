import numpy as np
from typing import List, Dict

from transformers import AutoModelForMaskedLM
from tokenizer import *
import logging

logger = logging.getLogger(__name__)

def format_predictions(predictions: np.array, vocab: Dict, rel_toks: List) -> List[str]:
    """
    Format the predictions of a model to a list of strings.

    :param predictions: The predictions of the model as a 1D numpy array.
    :param vocab: A dictionary mapping the indices to the labels.
    :param sentence_lengths: A list of integers representing the sentence lengths.
    :return: A list of strings representing the predictions.
    """
    formatted_output = []
    sent = ''
    prev_line_sent_ix = rel_toks[0].split('_')[0]

    for i, (label, tok) in enumerate(zip(predictions.tolist(), rel_toks)):
        current_idx = tok.split('_')[0]

        if current_idx != prev_line_sent_ix:
            formatted_output.append(sent + '\n')
            sent = f'{vocab[label]} '
            prev_line_sent_ix = current_idx
        else:
            sent += f'{vocab[label]} '
    formatted_output.append(sent)

    return ''.join(formatted_output)

def swap_labels(result, label_vocab):
    f_result = {}
    idx2c = {v: k for k, v in label_vocab.items()}

    # output of pytorch lightning .test is a list with all logged metrics, in this case only one dict
    for c, acc in result[0].items():
        if c == 'test_acc' or c == 'val_acc':
            f_result[c] = acc
        else:
            class_label = int(c.split('_')[1])
            f_result[idx2c[class_label]] = acc

    return f_result

def load_model(checkpoint, device):
    model = AutoModelForMaskedLM.from_pretrained(checkpoint).to(device)

    with open(f'{checkpoint}/added_tokens.json') as f:
        vocab = json.load(f)
    tokenizer = create_tf_tokenizer_from_vocab(vocab)

    return model, tokenizer