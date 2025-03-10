import torch
import logging
import sys
import numpy as np
from tqdm import tqdm

from data import Corpus
from utils.utils import get_all_subword_id, match_tokenized_to_untokenized
from utils.utils import load_model
from utils.wr_utils import write_im_to_file


logging.basicConfig(stream=sys.stdout,
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

def mask_ith_token(tokenized_text, i, mask_id, mapping):
    """
    Function to mask all the subwords of the i-th token.
    :param tokenized_text: list of tokenized text
    :param i: index of the token to be masked
    :param mask_id: id of the mask token
    :param mapping: mapping between tokenized text and original text

    :return: list of tokenized text with i-th token masked
    """
    id_for_all_i_tokens = get_all_subword_id(mapping, i)
    # mask i-th token. For each subword you need the indexed_tokens list again as a base.
    tmp_indexed_tokens = list(tokenized_text)
    for tmp_id in id_for_all_i_tokens:
        if mapping[tmp_id] != -1: # both CLS and SEP use -1 as id e.g., [-1, 0, 1, 2, ..., -1]
            tmp_indexed_tokens[tmp_id] = mask_id
            
    return tmp_indexed_tokens

def batch_mask_combinations(indexed_tokens, mask_id, mapping, tokenized_text):
    """
    Function to create a batch of masked tokens.
    :param indexed_tokens: list of tokenized text
    :param mask_id: id of the mask token
    :param mapping: mapping between tokenized text and original text
    :param tokenized_text: list of tokenized text

    :return: list of tokenized text with i-th token masked
    """
    one_batch = [list(indexed_tokens) for _ in range(0, len(tokenized_text))]
    for j in range(0, len(tokenized_text)):
        id_for_all_j_tokens = get_all_subword_id(mapping, j)
        for tmp_id in id_for_all_j_tokens:
            if mapping[tmp_id] != -1:
                one_batch[j][tmp_id] = mask_id

    return torch.tensor(one_batch)


def get_representations(model, tokens_tensor, config):
    """
    Function to get hidden states for all layers for one batch
    :param model: model
    :param tokens_tensor: tokens tensor
    :param segments_tensor: segments tensor
    :param config: configuration

    :return: hidden states for all layers
    """
    attention_mask = torch.ones_like(tokens_tensor).to(config.device)

    with torch.no_grad():
        all_layers = model(tokens_tensor, attention_mask=attention_mask, output_hidden_states=True).hidden_states

    return all_layers


def represenations_per_layer(all_layers, i, config):
    """
    Function to get hidden states for i-th token in all layers
    :param all_layers: hidden states for all layers
    :param i: index of the token

    :return: hidden states for i-th token in all layers
    """
    repr_per_layer = [[] for i in range(config.layers)]
    for k, layer in enumerate(all_layers):
        if config.device != 'cpu':
            hidden_states_for_token_i = layer[:, i, :].cpu().numpy()
        else:
            hidden_states_for_token_i = layer[:, i, :].numpy()
        repr_per_layer[k].append(hidden_states_for_token_i)

    return repr_per_layer 


def create_impact_matrices(config, all_layers_matrix_as_list, sents, tokenized_text, tree2list, nltk_tree):
    temp = [[] for i in range(config.layers)]
    for k, one_layer_matrix in enumerate(all_layers_matrix_as_list):
        init_matrix = np.zeros((len(tokenized_text), len(tokenized_text)))
        for i, hidden_states in enumerate(one_layer_matrix):
            base_state = hidden_states[i]
            for j, state in enumerate(hidden_states):
                if config.metric == 'dist':
                    init_matrix[i][j] = np.linalg.norm(base_state - state)
                if config.metric == 'cos':
                    init_matrix[i][j] = np.dot(base_state, state) / (
                            np.linalg.norm(base_state) * np.linalg.norm(state))
        temp[k].append((sents, tokenized_text, init_matrix, tree2list, nltk_tree))
    
    return temp


def prepare_sentence(sents, tokenizer):
    """
    Function to prepare sentence for analysis
    :param sentence: sentence
    :param tokenizer: tokenizer
    :param mask_id: mask id

    :return: tokenized text, indexed tokens, mapping
    """
    sentence = ' '.join(sents)
    tokenized_text = tokenizer.tokenize(sentence)
    tokenized_text.insert(0, '[CLS]')
    tokenized_text.append('[SEP]')
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    assert len(sents) == len(tokenized_text) - 2, 'Something went wrong with tokeninzing.'

    mapping = match_tokenized_to_untokenized(tokenized_text, sents)

    return tokenized_text, indexed_tokens, mapping

def extract_matrix(config, model, tokenizer, sents, tree2list, nltk_tree, mask_id):
    """
    Function to extract impact matrix
    :param model: model
    :param tokenizer: tokenizer
    :param sents: sentence
    :param tree2list: list version of tree
    :param nltk_tree: nltk tree
    :param mask_id: mask id
    """

    logger.debug('Tokenize sentences.')
    tokenized_text, indexed_tokens, mapping = prepare_sentence(sents, tokenizer)
    assert mapping != False, 'Tokenization mismatch'
    logger.debug('Tokenization done.')

    all_layers_matrix_as_list = [[] for i in range(config.layers)]

    for i in range(0, len(tokenized_text)):
        # 1. Mask i-th token
        tmp_indexed_tokens = mask_ith_token(indexed_tokens, i, mask_id, mapping)

        # i with all j gives one batch (sen length + 2, sen length + 2)
        one_batch = batch_mask_combinations(tmp_indexed_tokens, mask_id, mapping, tokenized_text).to(config.device)

        # 2. Get all hidden states for one batch
        all_layers = get_representations(model, one_batch, config)
        assert len(all_layers) == config.layers, f'Number of layers mismatch, {len(all_layers)} != {config.layers}'
        assert all_layers[0].shape[0] == len(sents) + 2, f'Batch size mismatch, {all_layers[0].shape[0]} != {len(sents) + 2}'

        # 3. get hidden states for word_i in one batch
        result_ith_word = represenations_per_layer(all_layers, i, config)

        for k, one_layer in enumerate(result_ith_word):
            all_layers_matrix_as_list[k].extend(one_layer)


    temp = create_impact_matrices(config, all_layers_matrix_as_list, sents, tokenized_text, tree2list, nltk_tree)

    return temp


def main_impact_matrix(config):
    """
    Main function to run span probing
    """
    logger.info('Creating impact matrices.')
    logger.info('Loading model...')
    model, tokenizer = load_model(config)
    model.eval()
    logger.info('Model loaded.')

    mask_id = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]

    corpus = Corpus(config.data, config.split)
    
    out = [[] for i in range(config.layers)]
    for sents, tree2list, nltk_tree in tqdm(zip(corpus.sens, corpus.trees, corpus.nltk_trees), total = len(corpus.sens)):
        per_sen_result = extract_matrix(config, model, tokenizer, sents, tree2list, nltk_tree, mask_id)

        for k, one_layer_result in enumerate(per_sen_result):
            out[k].extend(one_layer_result)

    write_im_to_file(out, config)

    return out