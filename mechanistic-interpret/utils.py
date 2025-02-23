import numpy as np
from typing import List, Dict

from transformers import AutoModelForMaskedLM, AutoModelForCausalLM
from tokenizer import *
import os
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

def load_model(checkpoint, device, type):
    if type == 'gpt2':
        model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
    elif type == 'deberta':
        model = AutoModelForMaskedLM.from_pretrained(checkpoint).to(device)

    with open(f'{checkpoint}/added_tokens.json') as f:
        vocab = json.load(f)
    tokenizer = create_tf_tokenizer_from_vocab(vocab)

    return model, tokenizer

def load_model_tokenizer(config_dict):
    if config_dict['model']['model_type'] == 'gpt2':
        automodel = AutoModelForCausalLM
    elif config_dict['model']['model_type'] == 'deberta':
        automodel = AutoModelForMaskedLM
    else:
        raise ValueError(f'Model not supported.')
    
    model = automodel.from_pretrained(config_dict['model']['model_file'])

    with open(f"{config_dict['model']['model_file']}added_tokens.json") as f:
        vocab = json.load(f)

    tokenizer = create_tf_tokenizer_from_vocab(vocab)

    return model, tokenizer


def set_experiment_config(args):
    # Check if model directory exists and find the correct checkpoint
    if not os.path.exists(args['model']['model_file']):
        raise ValueError(f"Model directory {args['model']['model_file']} does not exist.")
    
    highest_config = 0
    print(args['model']['model_file'])
    for dir_name in os.listdir(args['model']['model_file']):
        if dir_name.split('-')[0] == 'checkpoint':
            config = int(dir_name.split('-')[1])
            if config > highest_config:
                highest_config = config

    args['model']['model_file'] = f"{args['model']['model_file']}/checkpoint-{highest_config}/"

    return args

def store_model_probs(all_token_probs, datasets, fn: str):
    lines = []
    cur_idx = 0

    for sen in datasets['test']['text']:
        lines.append(sen)
        sen_len = len(sen.split(' ')) - 1

        for prob in all_token_probs[cur_idx:cur_idx+sen_len]:
            lines.append(str(prob))

        cur_idx += sen_len

    with open(fn, 'w') as f:
        f.write('\n'.join(lines))

