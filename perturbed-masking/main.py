import logging
import sys
import pprint
import pprint
import os
import pickle
from pathlib import Path

from extract_impact_matrix import main_impact_matrix
from argparser import create_arg_parser
from parsing import decoding
from evaluation import spearman_evaluation, classic_evaluation
from utils.wr_utils import listtree2str
from edge_probe import edge_probing_labelling


logging.basicConfig(stream=sys.stdout,
    level=logging.DEBUG,
    format='%(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)


def get_impact_matrices(config):
    all_layer_results = []
    exist = True
    # removing embedding layer
    for layer in range(config.layers):
        matrix_path = f'{config.output_dir}/{config.model}_{config.metric}_{layer}.pkl'

        if os.path.exists(matrix_path):
            # impact matrix
            with open(matrix_path, 'rb') as f:
                matrix = pickle.load(f)
            all_layer_results.append(matrix)
        else:
            exist = False

    if not exist:
        logger.info(f'File {matrix_path} does not exist. Generated now.')
        all_layer_results = main_impact_matrix(config)
    else:
        logger.info(f'Impact matrices already exist. Loaded from file system.')

    return all_layer_results


def get_predicted_trees(config, all_layer_info):
    pred_trees = []
    for i in range(config.layers):
        tree_file = Path(f'trees_{config.model}_{config.metric}_{i}_list.pkl') if config.evaluation == 'classic' or config.evaluation == 'labelled' \
            else Path(f'trees_{config.model}_{config.metric}_{i}.txt')
        tree_path = config.tree_path / tree_file

        if not tree_path.exists():
            logger.debug(f'File {tree_path} does not exist. Creating parse trees for layer {i}.')
            # TODO: fix this one for remove punctuation
            trees, _ = decoding(config, all_layer_info[i - 1])
            pred_trees.append(trees)

            if config.evaluation == 'classic' or config.evaluation == 'labelled':
                with open(tree_path, 'wb') as f:
                    pickle.dump(trees, f)
            elif config.evaluation == 'spearman':
                # convert trees to strings and write to file
                trees = [listtree2str(tree) for tree in trees]
                with open(tree_path, 'w') as f:
                    for tree in trees:
                        f.write(f'{tree}\n')

        else:
            logger.debug(f'File {tree_path} exists. Loading from file system.')
            # load parse trees
            if config.evaluation == 'classic' or config.evaluation == 'labelled':
                logger.debug('Loading trees as list.')
                with open(tree_path, 'rb') as f:
                    pred_trees.append(pickle.load(f))
            elif config.evaluation == 'spearman':
                logger.debug('Loading trees as strings.')
                with open(tree_path, 'r') as f:
                    pred_trees.append([line.strip('\n') for line in f.readlines()])
            else:
                logger.debug('Unknown evaluation type. As result, the correct way of saving is unknown. Exiting')
                raise ValueError(f'Evaluation type {config.evaluation} not supported.')

    logger.info('Parse trees loaded.')

    # if config.evaluation == 'classic' or config.evaluation == 'labelled':
    #     return pred_trees
    
    # elif config.evaluation == 'spearman':
    #     return [listtree2str(tree) for tree in trees]

    return pred_trees
    

def main(config):
    # 1. Load impact matrices
    # Contains: [(sentence, tokenized_text, impact matrix, tree2list, nltk_tree)]
    all_layer_info = get_impact_matrices(config)

    assert len(all_layer_info) == config.layers, f'Number of layers {len(all_layer_info)} does not match the expected number of layers {config.layers}.'
    # 2. Generate or load predicted trees
    pred_trees = get_predicted_trees(config, all_layer_info)

    # 2. Evaluate parse trees
    assert len(pred_trees) == config.layers, f'Number of layers {len(pred_trees)} does not match the expected number of layers {config.layers}.'
    print(config.evaluation, flush=True)
    if config.evaluation == 'spearman':
        logger.info('Started spearman evaluation...')
        spearman_evaluation(config, all_layer_info, pred_trees, split=config.split)
    elif config.evaluation == 'classic':
        logger.info('Started classic evaluation...')
        classic_evaluation(config, all_layer_info, pred_trees)
    elif config.evaluation == 'labelled':
        logger.info('Started labelled evaluation...')
        edge_probing_labelling(config, pred_trees)
    else:
        raise ValueError(f'Evaluation type {config.evaluation} not supported.')


if __name__ == '__main__':
    logger.info('Loading configuration...')
    config = create_arg_parser()
    logger.info('Configuration loaded.')
    pprint.pprint(vars(config))

    main(config)