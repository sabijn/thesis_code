import argparse
from pathlib import Path
import logging 
import torch
import os

logger = logging.getLogger(__name__)

def create_arg_parser():
    argparser = argparse.ArgumentParser(description='Perturbed Masking')

    # GENERAL
    
    # MODEL
    argparser.add_argument('--device', type=str, default=None,
                           help='Device to run on')
    argparser.add_argument('--model', type=str, default='deberta',
                           help='Model type')
    argparser.add_argument('--seed', type=int, default=42,
                           help='Random seed')

    # DATA OPTIONs
    argparser.add_argument('--embedding_layer', action=argparse.BooleanOptionalAction,
                           help='Include embedding layer in analysis.')
    
    # PATHS
    argparser.add_argument('--data', type=Path, default=Path('/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/corpora/eval_trees_10k.txt'),
                        help='Path to data')
    argparser.add_argument('--home_model_path', type=Path, default=Path('/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/pcfg-lm/resources/checkpoints/'),
                        help='Path to directory were the differen models are stored.')
    argparser.add_argument('--output_dir', type=Path, default=Path('/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/perturbed-masking/results/'),
                        help='Path to output')

    config = argparser.parse_args()

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
        model_path = config.home_model_path / Path('deberta/')
        layers = 8

    elif config.model == 'gpt2':
        model_path = config.home_model_path / Path('gpt2/')
        layers = 12 # TODO: check this
    
    if config.embedding_layer:
        layers += 1

    config.model_path = model_path
    config.layers = layers

    # Configure output files
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.output_file = config.output_dir / '{}-{}-{}-{}.pkl'

    return config