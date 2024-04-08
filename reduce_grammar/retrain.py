from classes import (PCFG, PCFGConfig,
                    TokenizerConfig, Tokenizer, 
                    ModelConfig, LanguageClassifier,
                    ExperimentConfig, Experiment)
from copy import deepcopy
from utils import plot_results
import argparse
import pickle
import torch
import logging

logger = logging.getLogger(__name__)

def main(args):
    if args.device is None:
        if torch.cuda.is_available():
            # For running on snellius
            args.device = torch.device("cuda")
            logger.info('Running on cuda.')
        elif torch.backends.mps.is_available():
            # For running on M1
            args.device = torch.device("mps")
            logger.info('Running on M1 GPU.')
        else:
            # For running on laptop
            args.device = torch.device("cpu")
            logger.info('Running on CPU.')

    grammar_file = f"{args.data_dir}/{args.version}/subset_pcfg_{args.top_k}.txt"
    encoder = 'transformer'

    logger.info('Loading tokenizer')
    tokenizer_config = TokenizerConfig(
        add_cls=(encoder == "transformer"),
        masked_lm=(encoder == "transformer"),
        unk_threshold=5,
    )
    pcfg_tokenizer = Tokenizer(tokenizer_config)

    logger.info('Initializing PCFG')
    config = PCFGConfig(
        is_binary=False,
        min_length=args.min_length,
        max_length=args.max_length,
        max_depth=100,
        corpus_size=args.corpus_size,
        grammar_file=grammar_file,
        start="S_0",
        masked_lm=(encoder == "transformer"),
        allow_duplicates=True,
        split_ratio=(.8,.1,.1),
        use_unk_pos_tags=True,
        file=args.corpus_file,
        device=args.device
    )

    lm_language = PCFG(config, pcfg_tokenizer)
    lm_language.save_pcfg()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retrain')
    parser.add_argument('--data_dir', type=str, default='grammars/nltk')
    parser.add_argument('--output_dir', type=str, default='corpora')
    parser.add_argument('--corpus_file', type=str, default=None)
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction)
    parser.add_argument('--corpus_size', type=int, default=1_000_000,
                        help='Size of the corpus to generate.')
    parser.add_argument('--min_length', type=int, default=6)
    parser.add_argument('--max_length', type=int, default=25)
    parser.add_argument('--max_depth', type=int, default=25)
    parser.add_argument('--top_k', type=float, default=0.2)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--version', type=str, default='normal', choices=['normal', 'pos', 'lexical'])

    args = parser.parse_args()

    main(args)





