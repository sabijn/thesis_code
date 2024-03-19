from evaluation import get_brackets
import torch
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

def create_data_splits(data):
    corpus_size = len(data)
    train_split, dev_split, test_split = int(0.8 * corpus_size), int(0.9 * corpus_size), corpus_size

    train_data = data[:train_split]
    dev_data = data[train_split:dev_split]
    test_data = data[dev_split:test_split]

    return train_data, dev_data, test_data


def create_span_ids(config, trees):
    spans_all_layers = []
    sentence_lengths = []
    for l in range(config.layers):
        spans_per_layer = []
        for tree in trees[l]:
            spans = list(get_brackets(tree)[0])
            sentence_lengths.append(get_brackets(tree)[1])
            spans.sort()

            spans = torch.tensor(spans)
            spans_per_layer.append(spans)
            
        spans_all_layers.append(spans_per_layer)
    
    return spans_all_layers, sentence_lengths


def load_states(config):
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
            all_states = torch.concat(all_states)
        # to be compatible with the for-loop over the layers
        all_states = {0: all_states}
    else:
        logger.info('States not found. Creating states.')
        raise NotImplementedError('States not found. Creating states.')


def edge_probing_labelling(config, trees):
    span_ids, sentence_lengths = create_span_ids(config, trees) # returns list of lists containing span tensors and length of sentences
    states = load_states(config)

    # use only the test set
    test_states = create_data_splits(states)[2]

    for l in range(config.layers):
        for i in range(len(test_states[l])):
            assert sentence_lengths[i] == test_states[l][i].shape[0]
        break

    # load probe model
    # def eval_probe(probe, states, spans, labels, label_vocab):
    # probe.eval()
    
    # all_labels = []
    # all_preds = []
    
    # for state, span, label in zip(states, spans, labels):
    #     with torch.no_grad():
    #         pred = probe(state.unsqueeze(0), span)
    #         pred = pred.argmax(-1)
            
    #         all_labels.append(label.tolist())

    # label vocab toevoegen (staat in edge-probing/data/label_vocab.json)
    



