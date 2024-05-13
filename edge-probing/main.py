import logging
import sys
import torch
import pprint
import nltk
from tqdm import tqdm
import pickle
import json
from pathlib import Path
import pprint

from argparser import create_arg_parser
from utils import load_model
from model import ProbeConfig
from data import extract_span_labels, create_states
from train import probe_loop

logging.basicConfig(stream=sys.stdout,
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

def main(config):
    """
    Main function to run span probing

    Edit for different top k models
    - Load model different
    - Paths different
    """
    logger.info('Running span probing.')
    logger.info('Loading model...')
    model, tokenizer = load_model(config.model_path, config.device)
    model.eval()
    logger.info('Model loaded.')

    logger.info('Loading data...')
    with open(config.data) as f:
        tree_corpus = [nltk.Tree.fromstring(l.strip()) for l in f]
    logger.info('Data loaded.')

    logger.info('Extracting span labels...')
    if (config.span_ids_path.exists() and 
        config.tokenized_labels_path.exists() and 
        config.label_vocab_path.exists()):

        logger.info('Span labels already extracted. Loading from filesystem.')
        all_span_ids = pickle.load(open(config.span_ids_path, 'rb'))
        all_tokenized_labels = pickle.load(open(config.tokenized_labels_path, 'rb'))
        span_label_vocab = json.load(open(config.label_vocab_path, 'r'))
    else:
        all_span_ids, all_tokenized_labels, span_label_vocab = extract_span_labels(
            tree_corpus, 
            device=config.device,
            merge_pos=True,
            merge_at=True,
            skip_labels=config.skip_labels,
        )
        pickle.dump(all_span_ids, open(config.span_ids_path, 'wb'))
        pickle.dump(all_tokenized_labels, open(config.tokenized_labels_path, 'wb'))
        json.dump(span_label_vocab, open(config.label_vocab_path, 'w'))

    logger.info('Span labels extracted.')

    logger.info('Creating states...')
    if Path(config.created_states_path / 'states_all_layers.pkl').exists() and config.all_layers:
        logger.info('States already created. Loading from filesystem with all layers.')
        all_states = pickle.load(open(config.created_states_path / Path('states_all_layers.pkl'), 'rb'))
        if config.concat:
            all_states = {layer: torch.concat(states) for layer, states in all_states.items()}

    elif Path(config.created_states_path / 'states.pkl').exists() and not config.all_layers:
        logger.info('States already created. Loading from filesystem with only final layer.')
        all_states = pickle.load(open(config.created_states_path / Path('states.pkl'), 'rb'))
        if config.concat:
            all_states = torch.concat(states)
        # to be compatible with the for-loop over the layers
        all_states = {0: all_states}
    else:
        all_states = create_states(tokenizer, tree_corpus, model, config, concat=config.concat, verbose=True, all_layers=config.all_layers)  
        logger.info('States created.')

    layer_f1 = []
    layer_predictions = []
    # train probes per layer activation
    for idx, states in enumerate(tqdm(all_states.values())):
        hidden_size = states[0].shape[-1]

        probe_config = ProbeConfig(
            lr=config.lr,
            epochs=config.epochs,
            verbose=config.verbose,
            batch_size=config.batch_size,
            weight_decay=config.weight_decay,
            device=config.device
        )

        span_probe, loss_curve, train_accs, dev_accs, test_merged_f1, test_preds = probe_loop(
            states, 
            all_span_ids, 
            all_tokenized_labels, 
            hidden_size, 
            span_label_vocab,
            probe_config
        )
        
        layer_f1.append(test_merged_f1)
        layer_predictions.append(test_preds)
    
    with open(config.output_path / 'results_f1_all_layers.pkl', 'wb') as f:
        pickle.dump(layer_f1, f)
    
    with(open(config.output_path / 'predictions_all_layers.pkl', 'wb')) as f:
        pickle.dump(layer_predictions, f)
    
    logger.info('Span probing done. Results are in data/results_f1_all_layers.pkl')

if __name__ == '__main__':
    """
    Code to run span probing
    """
    logger.info('Loading configuration...')
    config = create_arg_parser()
    logger.info('Configuration loaded.')
    pprint.pprint(vars(config))

    main(config)