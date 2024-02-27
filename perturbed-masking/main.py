# import os
# import pickle
# import argparse
# from tqdm import tqdm
# import numpy as np
# from utils import ConllUDataset
# from transformers import BertModel, BertTokenizer
# from dependency import get_dep_matrix
# from constituency import get_con_matrix
# from discourse import get_dis_matrix

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
import numpy as np
import os

from utils import load_model, write_to_file
from argparser import create_arg_parser
from data import Corpus
from extract_impact_matrix import extract_matrix

logging.basicConfig(stream=sys.stdout,
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)


def main(config):
    """
    Main function to run span probing
    """
    logger.info('Running span probing.')
    logger.info('Loading model...')
    model, tokenizer = load_model(config)
    model.eval()
    logger.info('Model loaded.')

    mask_id = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
    print(mask_id)

    corpus = Corpus(config.data)
    
    out = [[] for i in range(config.layers)]
    for sents, tree2list, nltk_tree in tqdm(zip(corpus.sens, corpus.trees, corpus.nltk_trees), total = len(corpus.sens)):
        per_sen_result = extract_matrix(config, model, tokenizer, sents, tree2list, nltk_tree, mask_id)

        for k, one_layer_result in enumerate(per_sen_result):
            out[k].extend(one_layer_result)

    write_to_file(out, config)

if __name__ == '__main__':
    logger.info('Loading configuration...')
    config = create_arg_parser()
    logger.info('Configuration loaded.')
    pprint.pprint(vars(config))

    main(config)


