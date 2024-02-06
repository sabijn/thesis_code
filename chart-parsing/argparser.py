import argparse
from pathlib import Path
import logging 
import torch

logger = logging.getLogger(__name__)

def create_arg_parser():
    argparser = argparse.ArgumentParser(description='Chart parsing')

    # GENERAL
    argparser.add_argument('--verbose', action=argparse.BooleanOptionalAction,
                        help='Verbose')
    
    # MODEL
    argparser.add_argument('--batch_size', type=int, default=48,
                           help='Batch size')
    argparser.add_argument('--device', type=str, default=None,
                           help='Device to run on')
    argparser.add_argument('--epochs', type=int, default=10,
                            help='Number of epochs')
    argparser.add_argument('--lr', type=float, default=10e-3, 
                           help='Learning rate')
    argparser.add_argument('--model', type=str, default='deberta',
                           help='Model type')
    argparser.add_argument('--seed', type=int, default=42,
                           help='Random seed')
    argparser.add_argument('--weight_decay', type=float, default=0.1,
                           help='Weight decay')

    # DATA OPTIONs
    # even over nadenken
    group = argparser.add_mutually_exclusive_group()
    group.add_argument("--all_layers", action='store_const', dest='all_layers', const=True,
                        help='All layers')
    group.add_argument("--only_final_layer", action='store_const', dest='all_layers', const=False,
                        help='Only final layer')
    group.set_defaults(all_layers=True)

    argparser.add_argument('--concat', action=argparse.BooleanOptionalAction,
                            help='Concat')
    argparser.add_argument('--skip_labels', type=list, default=['WHPP', 'NX', 'WHNP', 'X', 'WHADJP', 'WHADVP', 'RRC', 'NAC', 'CONJP', 'SBARQ'],
                            help='Skip labels')
    
    # PATHS
    argparser.add_argument('--created_states_path', type=Path, default=Path('/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/chart-parsing/data/'),
                            help='Path to created states')
    argparser.add_argument('--data', type=Path, default=Path('/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/corpora/eval_trees_10k.txt'),
                        help='Path to data')
    argparser.add_argument('--home_model_path', type=Path, default=Path('/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/pcfg-lm/resources/checkpoints/'),
                        help='Path to directory were the differen models are stored.')
    argparser.add_argument('--label_vocab_path', type=Path, default=Path('/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/chart-parsing/data/label_vocab.json'),
                            help='Path to label vocabulary')
    argparser.add_argument('--output_path', type=Path, default=Path('/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/chart-parsing/results/'),
                        help='Path to output')
    argparser.add_argument('--span_ids_path', type=Path, default=Path('/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/chart-parsing/data/span_ids.pkl'),
                            help='Path to span ids')
    argparser.add_argument('--tokenized_labels_path', type=Path, default=Path('/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/chart-parsing/data/tokenized_labels.pkl'),
                            help='Path to tokenized labels')
    
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