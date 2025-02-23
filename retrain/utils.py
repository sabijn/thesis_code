import json
from transformers import AutoModelForMaskedLM, AutoModelForCausalLM
import os
from collections import defaultdict
import numpy as np
import pickle

from tokenizer import create_tf_tokenizer_from_vocab

def set_experiment_config(args):
    model_dir = f'{args.model_dir}/{args.model}/{args.version}/{args.top_k}'

    # Check if model directory exists and find the correct checkpoint
    if not os.path.exists(model_dir):
        raise ValueError(f'Model directory {model_dir} does not exist.')
    
    highest_config = 0
    for dir_name in os.listdir(model_dir):
        if dir_name.split('-')[0] == 'checkpoint':
            config = int(dir_name.split('-')[1])
            if config > highest_config:
                highest_config = config

    args.model_file = f'{model_dir}/checkpoint-{highest_config}/'

    # Check if data directory exist and construct the directory corresponding to the current version
    # args.data_dir_comp = f'{args.data_dir}/{args.version}'

    if not os.path.exists(args.data_dir):
        raise ValueError(f'Data directory {args.data_dir} does not exist.')

    return args

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

def get_model_prob_dict(fn):
    with open(fn) as f:
        lines = f.read().split('\n')

    sen2lm_probs = defaultdict(list)
    cur_sen = None

    for line in lines:
        if len(line) == 0:
            continue
        if not line[-1].isnumeric():
            cur_sen = line
        else:
            sen2lm_probs[cur_sen].append(float(line))

    return sen2lm_probs


def get_pcfg_prob_dict(filename='lm_training/eval_sens_probs.txt'):
    with open(filename) as f:
        lines = f.read().split('\n')

    sen2pcfg_probs = defaultdict(list)
    cur_sen = None

    for line in lines:
        if len(line) == 0:
            continue
        if line.startswith('Token:'):
            continue
        elif not line[-1].isnumeric():
            cur_sen = line
        else:
            sen2pcfg_probs[cur_sen].append(float(line))
            
    return sen2pcfg_probs


def store_model_probs(all_token_probs, datasets, fn: str):
    lines = []
    cur_idx = 0

    for sen in datasets['test']['text'][:100]:
        lines.append(sen)
        sen_len = len(sen.split(' '))

        for prob in all_token_probs[cur_idx:cur_idx+sen_len]:
            lines.append(str(prob))

        cur_idx += sen_len

    with open(fn, 'w') as f:
        f.write('\n'.join(lines))


# def get_probs(datasets, tokenizer, sen2lm_probs, sen2pcfg_probs):
#     lm_probs = []
#     pcfg_probs = []
#     all_tokens = []
    
#     skip_tokens = {tokenizer.unk_token_id}

#     for sen, input_ids in zip(datasets['test']['text'], datasets['test']['input_ids']):
#         pcfg_sen = sen.replace("<apostrophe>", "'")

#         sen_pcfg_probs = sen2pcfg_probs[pcfg_sen]
#         sen_lm_probs = sen2lm_probs[sen]

#         assert len(sen_pcfg_probs) == len(sen_lm_probs), f"{sen}\n{len(sen_pcfg_probs)},{len(sen_lm_probs)}"

#         iterator = zip(input_ids, sen_lm_probs, sen_pcfg_probs)
#         for idx, (input_id, lm_prob, pcfg_prob) in enumerate(iterator):
#             if input_id not in skip_tokens:
#                 lm_probs.append(np.exp(lm_prob))
#                 pcfg_probs.append(pcfg_prob)
#                 all_tokens.append(f"{idx}__{sen}")


#     lm_probs = np.array(lm_probs)
#     pcfg_probs = np.array(pcfg_probs)

#     lm_probs = np.log(lm_probs)
#     pcfg_probs = np.log(pcfg_probs)

#     return lm_probs, pcfg_probs, all_tokens

def get_probs(datasets, tokenizer, sen2lm_probs):
    lm_probs = []
    all_tokens = []
    
    skip_tokens = {tokenizer.unk_token_id}

    for sen, input_ids in zip(datasets['test']['text'], datasets['test']['input_ids']):
        pcfg_sen = sen.replace("<apostrophe>", "'")
        sen_lm_probs = sen2lm_probs[sen]

        iterator = zip(input_ids, sen_lm_probs)
        for idx, (input_id, lm_prob) in enumerate(iterator):
            if input_id not in skip_tokens:
                lm_probs.append(np.exp(lm_prob))
                all_tokens.append(f"{idx}__{sen}")

    lm_probs = np.array(lm_probs)

    lm_probs = np.log(lm_probs)

    return lm_probs, all_tokens


def get_causal_lm_pcfg_probs(pcfg_dict_fn, all_sen_probs, corpus, tokenizer):

    with open(pcfg_dict_fn) as f:
        pcfg_dict = json.load(f)

    pcfg_probs = []
    lm_probs = []

    for sen, pcfg_sen_probs in pcfg_dict.items():
        sen_idx = corpus.index(sen)
        lm_sen_probs = all_sen_probs[sen_idx]

        lm_probs.extend(lm_sen_probs)

        for idx, (w, prob) in enumerate(zip(sen.split(), pcfg_sen_probs)):
            if idx > 0 and w in tokenizer.vocab:
                pcfg_probs.append(-prob)
          
    return pcfg_probs, lm_probs