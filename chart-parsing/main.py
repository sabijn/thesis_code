import logging
import sys
import torch
import pprint
import nltk
import tqdm

from argparser import create_arg_parser
from utils import load_model
from model import ProbeConfig
from data import extract_span_labels, create_states
from train import probe_loop

logging.basicConfig(stream=sys.stdout,
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

def main(config, model, tokenizer):
    """
    Main function to run span probing
    """
    logger.info('Running span probing...')

    skip_labels = ['WHPP', 'NX', 'WHNP', 'X', 'WHADJP', 'WHADVP', 'RRC', 'NAC', 'CONJP', 'SBARQ']

    logger.info('Loading data...')
    with open(config.data) as f:
        tree_corpus = [nltk.Tree.fromstring(l.strip()) for l in f]
    logger.info('Data loaded.')

    all_span_ids, all_tokenized_labels, span_label_vocab = extract_span_labels(
        tree_corpus, 
        merge_pos=True,
        merge_at=True,
        skip_labels=skip_labels,
    )
    all_states = create_states(tokenizer, tree_corpus, model, concat=False, verbose=True, all_layers=True)  

    layer_f1 = []
    for states in tqdm(all_states.values()):
        hidden_size = states[0].shape[-1]

        probe_config = ProbeConfig(
            lr=config.lr,
            epochs=config.epochs,
            verbose=True,
            batch_size=config.batch_size,
            weight_decay=config.weight_decay
        )

        span_probe, loss_curve, train_accs, dev_accs, test_merged_f1, test_preds = probe_loop(
            states, 
            all_span_ids, 
            all_tokenized_labels, 
            hidden_size, 
            span_label_vocab,
            probe_config,
        )
        
        layer_f1.append(test_merged_f1)


if __name__ == '__main__':
    """
    Code to run span probing

    What to do:
    - self attentitive ding aan jaap vragen
    - device in data
    - 
    """
    logger.info('Loading configuration...')
    config = create_arg_parser()
    logger.info(f'Running span probing with the following configuration: {config}')

    logger.info('Loading model...')
    model, tokenizer = load_model(config.model_path, config.device)
    model.eval()
    logger.info('Model loaded.')

    main(config, model, tokenizer)