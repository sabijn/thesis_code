import torch
import logging
import sys

from utils import get_all_subword_id, match_tokenized_to_untokenized
import numpy as np

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

    return one_batch


def get_representations(model, tokens_tensor, segments_tensor):
    """
    Function to get hidden states for all layers for one batch
    :param model: model
    :param tokens_tensor: tokens tensor
    :param segments_tensor: segments tensor
    :param config: configuration

    :return: hidden states for all layers
    """
    with torch.no_grad():
        model_outputs = model(tokens_tensor, segments_tensor)
        all_layers = model_outputs[-1]  # 12 layers + embedding layer

    return all_layers


def represenations_per_layer(all_layers, i, config):
    """
    Function to get hidden states for i-th token in all layers
    :param all_layers: hidden states for all layers
    :param i: index of the token

    :return: hidden states for i-th token in all layers
    """
    all_layers_matrix_as_list = []
    for layer in all_layers:
        if config.device != 'cpu':
            hidden_states_for_token_i = layer[:, i, :].cpu().numpy()
        else:
            hidden_states_for_token_i = layer[:, i, :].numpy()
        all_layers_matrix_as_list.append(hidden_states_for_token_i)

    return all_layers_matrix_as_list


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


def prepare_sentence(sents, tokenizer, mask_id):
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

    mapping = match_tokenized_to_untokenized(tokenized_text, sentence)

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
    tokenized_text, indexed_tokens, mapping = prepare_sentence(sents, tokenizer, mask_id)
    assert mapping != False, 'Tokenization mismatch'
    logger.debug('Tokenization done.')

    all_layers_matrix_as_list = [[] for i in range(config.layers)]
    for i in range(0, len(tokenized_text)):
        # 1. Mask i-th token
        tmp_indexed_tokens = mask_ith_token(indexed_tokens, i, mask_id, mapping)
        one_batch = batch_mask_combinations(tmp_indexed_tokens, mask_id, mapping, tokenized_text)

        # 2. Convert one batch to PyTorch tensors
        tokens_tensor = torch.tensor(one_batch).to(config.device)
        segments_tensor = torch.tensor([[0 for _ in one_sent] for one_sent in one_batch]).to(config.device)

        # 3. Get all hidden states for one batch
        all_layers = get_representations(model, tokens_tensor, segments_tensor)

        # 4. get hidden states for word_i in one batch
        all_layers_matrix_as_list = represenations_per_layer(all_layers, i, config)

    temp = create_impact_matrices(config, all_layers_matrix_as_list, sents, tokenized_text, tree2list, nltk_tree)

    return temp