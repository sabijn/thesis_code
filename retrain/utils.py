import json
from transformers import AutoModelForMaskedLM, AutoModelForCausalLM
import os

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
    args.data_dir = f'{args.data_dir}/{args.version}'

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