import argparse
from pathlib import Path
import logging 
import torch
import os

logger = logging.getLogger(__name__)

def set_experiment_config(args, model_dir):
    # Check if model directory exists and find the correct checkpoint
    if not os.path.exists(model_dir):
        raise ValueError(f'Model directory {model_dir} does not exist.')
    
    highest_config = 0
    for dir_name in os.listdir(model_dir):
        if dir_name.split('-')[0] == 'checkpoint':
            config = int(dir_name.split('-')[1])
            if config > highest_config:
                highest_config = config

    model_file = f'{model_dir}/checkpoint-{highest_config}/'

    return model_file

def create_arg_parser():
    argparser = argparse.ArgumentParser(description='Perturbed Masking')

    # GENERAL
    argparser.add_argument('--metric', type=str, default='dist', choices=['dist', 'cos'],
                           help='Metric to use for analysis')
    argparser.add_argument('--quiet', action=argparse.BooleanOptionalAction,
                           help='Print verbose information')
    argparser.add_argument('--remove_punct', action=argparse.BooleanOptionalAction)
    
    # MODEL
    argparser.add_argument('--device', type=str, default=None,
                           help='Device to run on')
    argparser.add_argument('--model', type=str, default='deberta', choices=['deberta', 'gpt2'],
                           help='Model type')
    argparser.add_argument('--seed', type=int, default=42,
                           help='Random seed')

    # DATA OPTIONs
    argparser.add_argument('--embedding_layer', action=argparse.BooleanOptionalAction,
                           help='Include embedding layer in analysis.')
    argparser.add_argument('--top_k', type=float, default=0.2,
                           help='Top k percentage of tokens to perturb')
    argparser.add_argument('--version', type=str, default='normal')
    
    # PATHS
    argparser.add_argument('--data', type=Path, default=Path('/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/corpora/eval_trees_10k.txt'),
                        help='Path to data')
    argparser.add_argument('--home_model_path', type=Path, default=Path('/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/pcfg-lm/resources/checkpoints/'),
                        help='Path to directory were the differen models are stored.')
    argparser.add_argument('--output_dir', type=Path, default=Path('/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/perturbed-masking/test_results/i_matrices/'),
                        help='Path to output')
    argparser.add_argument('--tree_path', type=Path, default=Path('/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/perturbed-masking/test_results/'),
                        help='Path to directory were the trees are stored.')
    argparser.add_argument('--span_data', type=Path, default=Path('/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/perturbed-masking/span_data/'))
    argparser.add_argument('--created_states_path', type=Path, default=Path('/Users/sperdijk/Documents/Master/Thesis/thesis_code/edge-probing/data/'))
    argparser.add_argument('--eval_results_dir', type=Path, default='/results/eval/')
    
    # DECODER 
    argparser.add_argument('--decoder', default='mart')
    argparser.add_argument('--subword', default='avg')

    # EVALUATION
    argparser.add_argument('--evaluation', default='spearman', choices=['spearman', 'classic', 'labelled'],
                           help='Type of evaluation to perform')
    argparser.add_argument('--all_layers', action=argparse.BooleanOptionalAction)
    argparser.add_argument('--concat', action=argparse.BooleanOptionalAction)
    argparser.add_argument('--split', default=1.0, type=float,
                           help='Split of the data for testing')

    config = argparser.parse_args()

    if config.split > 1.0:
        config.split = int(config.split)

    # Configure device
    if config.device is None:
        if torch.cuda.is_available():
            # For running on snellius
            device = torch.device("cuda")
            logger.info('Running on GPU.')
        elif torch.backends.mps.is_available():
            # For running on M1
            device = torch.device("mps")
            logger.info('Running on M1 GPU.')
        else:
            # For running on laptop
            device = torch.device("cpu")
            logger.info('Running on CPU.')
    else:
        device = torch.device(config.device)

    config.device = device

    # Configure model
    if config.model == 'deberta':
        model_path = config.home_model_path / Path('babyberta/')
        layers = 8

    elif config.model == 'gpt2':
        model_path = config.home_model_path / Path('gpt2/')
        layers = 8

    if config.version and config.top_k:
        model_path = model_path / Path(f'{config.version}/{config.top_k}/')
        model_path = set_experiment_config(config, model_path)

    if config.embedding_layer:
        layers += 1
    
    config.model_path = model_path
    config.layers = layers

    # Configure output files
    config.output_dir = config.output_dir / Path(f'{config.model}') / Path(f'{config.version}') / Path(f'{config.top_k}')
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Configure tree results path
    if config.remove_punct:
        config.tree_path = config.tree_path / Path('trees_without_punct/')
    else:
        config.tree_path = config.tree_path / Path('trees/')
    
    if config.version and config.top_k:
        config.tree_path = config.tree_path / Path(f'{config.model}/{config.version}/{config.top_k}/')
    config.tree_path.mkdir(parents=True, exist_ok=True)

    # Configure evaluation path
    config.eval_results_dir = config.eval_results_dir / Path(f'{config.model}/{config.version}/{config.top_k}/')
    config.eval_results_dir.mkdir(parents=True, exist_ok=True)

    # Configure input data
    if config.version and config.top_k:
        config.data = config.data / Path(f'{config.version}/all_trees_{config.version}_{config.top_k}.txt')

    return config