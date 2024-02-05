import argparse
from pathlib import Path
import logging 
import torch

logger = logging.getLogger(__name__)

def create_arg_parser():
    argparser = argparse.ArgumentParser(description='Chart parsing')
    argparser.add_argument('--lr', type=float, default=10e-3, 
                           help='Learning rate')
    argparser.add_argument('--epochs', type=int, default=10,
                            help='Number of epochs')
    argparser.add_argument('--batch_size', type=int, default=48,
                           help='Batch size')
    argparser.add_argument('--device', type=str, default=None,
                           help='Device to run on')
    argparser.add_argument('--seed', type=int, default=42,
                           help='Random seed')
    argparser.add_argument('--data', type=Path, default=Path('/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/corpora/eval_trees_10k.txt'),
                           help='Path to data')
    argparser.add_argument('--model', type=str, default='deberta',
                           help='Model type')
    argparser.add_argument('--home_model_path', type=Path, default=Path('/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/pcfg-lm/resources/checkpoints/'),
                           help='Path to directory were the differen models are stored.')
    argparser.add_argument('--weight_decay', type=float, default=0.1,
                           help='Weight decay')
    
    config = argparser.parse_args()

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

    if config.model == 'deberta':
        model_path = config.home_model_path / Path('deberta/')
    elif config.model == 'gpt2':
        model_path = config.home_model_path / Path('gpt2/')
    
    config.model_path = model_path

    return config