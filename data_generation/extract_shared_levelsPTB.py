from treetoolbox import find_node, load_ptb, np_regex, vp_regex, tr_leaf_regex, find_end_indices, find_xps_above_i, address_is_xp, find_label, find_tracing_alignment, find_trace_ix, preprocess, lowest_phrase_above_leaf_i
# during test: from data_prep.treetoolbox
import sys
import argparse
import re
import random
import numpy as np
import nltk
from tqdm import tqdm
from utils import load_model
from pathlib import Path
import torch
import os

punct_regex = re.compile(r"[^\w][^\w]?")

def specialConditionNPEmbed(tree, i, j, emb_depth=None):
    """special condition that i and j are leaf indices in the tree, and that they are a token pair such that
    (i) the LCA is an NP and 
    (ii) the right token has more NPs above it than the left (meaning the right token is deeper embedded than the left)
    """
    xps_above_i = len(find_xps_above_i(i, tree, np_regex))
    xps_above_j = len(find_xps_above_i(j, tree, np_regex))
    if emb_depth is None:
        return xps_above_i < xps_above_j
    else:
        # print(xps_above_j - xps_above_i)
        return xps_above_j - xps_above_i == emb_depth

def phrasePropertiesForAdjacentTokenPairsInPTB(k, tree, tokenizer, skip_unkown_tokens=True):
    """
    Find LCA for Adjacent Token Pairs
    """
    shared_levels = []
    shared_only_root = []
    sent = tree.leaves()

    if tree.label() == 'S_0' and len(tree) == 1:
        tree = tree[0]

    # loop thourgh tokens
    for i in range(len(sent) - 1):
        i_tok = sent[i]

        # if i not in tokenizer.vocab, skip this word
        if skip_unkown_tokens:
            if i_tok not in tokenizer.vocab:
                continue

        j = i + 1
        j_tok = sent[j]

        # if j not in tokenizer.vocab, skip this word and calculate LCA with next token
        if skip_unkown_tokens:
            while j_tok not in tokenizer.vocab:
                j = j + 1
                j_tok = sent[j]

        lowest_common_ancestor = tree.treeposition_spanning_leaves(i,j+1)

        # If LCA is the root, append index of token pair to shared_only_root
        if len(lowest_common_ancestor) == 0:
            shared_only_root.append(len(shared_levels))

        # Depth in tree shared between i and j
        shared = len(lowest_common_ancestor) + 1
        shared_levels.append(shared)

    # Calculate relative shared levels
    shared_levels_rel = []
    for l, s in enumerate(shared_levels):
        if l == 0:
            # If first token, append shared level of token 0 & 1
            # if s < -4:
            #     s = -4

            # elif s > 4:
            #     s = 4

                
            shared_levels_rel.append(str(s)) 
            last_s = s
        else:
            rel = s - last_s
            # if rel < -4:
            #     rel = -4
            # elif rel > 4:
            #     rel = 4

            shared_levels_rel.append(str(rel))
            last_s = s
            
    shared_levels_rel = []
    for l, s in enumerate(shared_levels):
        if l == 0:
            # If first token, append shared level of token 0 & 1
            shared_levels_rel.append(str(s)) 
            last_s = s
        else:
            shared_levels_rel.append(str(s - last_s))
            last_s = s
    return shared_levels_rel
    # shared_outliers = []
    # for s in shared_levels_rel:
    #     if int(s) < -4 or int(s) > 4:
    #         shared_outliers.append(s)
    #     else:
    #         shared_outliers.append(str(0))

    # return shared_outliers

def sample_data():
    pass

if __name__=='__main__':
    """
    Usage: 
    python3 span_prediction_format -ptb_tr <file_pattern> -ptb_notr <file_pattern> -text_toks <filename> -rel_toks <filename> -rel_labels <filename>

    Input Options: 
    - PTB with and without traces (both are needed)

    Output Options (each file has same number of lines): 
    -text_toks txt file with one sentence per line (input file for computing activations)
    -rel_toks txt file with tokens i_j for each i, j within the sentence length
    -rel_labels txt file where the k-th label in the l-th line represents the lowest common ancestor label in the PTB of the corresponding tokens in rel_toks.txt
    -max_span_const optional file. If present, binary labels are printed to the file that indicate whether or not the token pair is the first and last element of a constituent or not

    other options
    -cutoff x an integer x such that the scripts stops after processing x sentences
    -max_sent_length an integer x for the maximum sentence length (suggested: 20)
    -np_embed special condition for sampling only output labels such that 
    The script asserts that all output files have the same number of lines, and that the output files rel_toks and rel_labels have the same number of elements per line
    -next create only output for adjacent tokens, instead of all possible token pairs. 
    -shared_levels is used together with next. It indicates for a pair of adjacent tokens the variation in the number of shared tree levels between adjacent tokens (i.e. how much deeper in the tree is the LCA of the current token pair, compared to the last token pair?)
    -unary is used together with next. It indicates the output file with labels of unary leaf chains above a token.
    """

    """
    Call with: python extract_shared_levels.py
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=Path('/Users/sperdijk/Documents/Master/Jaar_3/Thesis/PTB/penn-line.txt'))
    parser.add_argument('--rel_toks', default=Path('/Users/sperdijk/Documents/Master/Jaar_3/Thesis/PTB/penn-sentences.txt'))
    parser.add_argument('--rel_labels') 
    parser.add_argument('--cutoff')
    parser.add_argument('--max_sent_length', default=31)
    parser.add_argument('--next', action='store_true', default=True) # run in experiments
    parser.add_argument('--shared_levels', default=Path('/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/data/shared_levels_PTB.txt')) # run in experiments
    parser.add_argument('--unary', default=None) # run in experiments
    parsedargs = parser.parse_args()

    ignored_sents = [] # added if not length: 3 < length < 31
    ignore_list = [] # no idea but this is in the original code

    if torch.cuda.is_available():
        # For running on snellius
        device = torch.device("cuda")
        print('Running on GPU.')
    # elif torch.backends.mps.is_available():
    #     # For running on M1
    #     device = torch.device("mps")
    #     print('Running on M1 GPU.')
    else:
        # For running on laptop
        device = torch.device("cpu")
        print('Running on CPU.')

    home_path = Path(os.environ['CURRENT_WDIR'])
    model_path = home_path / Path('pcfg-lm/resources/checkpoints/deberta/')
    _, tokenizer = load_model(model_path, device)

    # Set thresholds
    cutoff = np.inf
    if parsedargs.cutoff is not None:
        cutoff = int(parsedargs.cutoff)
    max_sent_length = np.inf
    if parsedargs.max_sent_length is not None:
        max_sent_length = int(parsedargs.max_sent_length)

    # Reading in input trees
    with open(home_path / parsedargs.data) as f:
        tree_corpus = [nltk.Tree.fromstring(l.strip()) for l in f]

    # Open output files
    shared_levels_file = open(parsedargs.shared_levels, 'w')

    binary = False
    
    ###############
    # Conversion 
    ###############
    for k, tree in enumerate(tqdm(tree_corpus)):
        if k - len(ignored_sents) > cutoff:
            continue
        # some sentences are not covered yet
        if k in ignore_list or len(tree.leaves()) > max_sent_length: #31:
            ignored_sents.append(k)
            continue

        next = parsedargs.next
        if next:
            if len(tree.leaves()) < 3:
                ignored_sents.append(k)
                continue

            shared_levels_rel = phrasePropertiesForAdjacentTokenPairsInPTB(k - len(ignored_sents), tree, tokenizer)

        shared_levels_file.write(' '.join(shared_levels_rel) + '\n')

    shared_levels_file.close()
    print('finished. ignored sentences: ', ignored_sents)
