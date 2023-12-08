from treetoolbox import find_node, load_ptb, np_regex, vp_regex, tr_leaf_regex, find_end_indices, find_xps_above_i, address_is_xp, find_label, find_tracing_alignment, find_trace_ix, preprocess, lowest_phrase_above_leaf_i
# during test: from data_prep.treetoolbox
import sys
import argparse
import re
import random
import numpy as np
import nltk
from tqdm import tqdm

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

def phrasePropertiesForAdjacentTokenPairsInPTB(k, tree):
    """
    Find LCA for Adjacent Token Pairs
    """
    rel_toks = []
    lca_labels = []
    max_span_labels = []
    shared_levels = []
    unary_labels = []
    shared_only_root = []
    sent = tree.leaves()

    if tree.label() == 'S_0' and len(tree) == 1:
        tree = tree[0]

    # loop thourgh tokens
    for i in range(len(sent) - 1):
        i_tok = sent[i]

        j = i + 1
        j_tok = sent[j]

        lowest_common_ancestor = tree.treeposition_spanning_leaves(i,j+1)

        # If LCA is the root, append index of token pair to shared_only_root
        if len(lowest_common_ancestor) == 0:
            shared_only_root.append(len(shared_levels))

        # Depth in tree shared between i and j
        shared = len(lowest_common_ancestor) + 1
        shared_levels.append(shared)

        # Get label of LCA (otherwise tuple representation)
        label = tree[lowest_common_ancestor].label()

        # check for unary leaves
        i_tok_address = tree.leaf_treeposition(i)
        unary_label = 'XX'
        above_pos = tree[i_tok_address[:-2]]
        if (len(above_pos) == 1):
            unary_label = above_pos.label()
        
        unary_label = re.sub('[^A-Za-z]+', '', unary_label)
        label = re.sub('[^A-Za-z]+', '', label)

        lca_labels.append(label)
        rel_toks.append('_'.join([str(k), str(i), str(j)]))
        unary_labels.append(unary_label)
        max_span_labels.append('0')

    # Calculate relative shared levels
    shared_levels_rel = []
    for l, s in enumerate(shared_levels):
        if l == 0:
            # If first token, append shared level of token 0 & 1
            shared_levels_rel.append(str(s)) 
            last_s = s
        else:
            shared_levels_rel.append(str(s - last_s))
            last_s = s

    # If ROOT is the only shared level, replace int with ROOT
    for r in shared_only_root:
        shared_levels_rel[r] = "ROOT"

    return rel_toks, lca_labels, shared_levels_rel, unary_labels


# def phrasePropertiesForTokenPairsInPTB(k, sent, tree, specialCondition='', emb_depth=None): 
#     rel_toks = []
#     lca_labels = []
#     max_span_labels = []

#     for i in range(len(sent)):
#         i_tok = sent[i]

#         if punct_regex.match(i_tok):
#             continue
#         if not(specialCondition == 'NP_embed'):
#             # find phrase above token i

#             label, node = lowest_phrase_above_leaf_i(i, tree)
#             label = label.split('-')[0]

#             max_span_label = '1' if len(node.leaves()) == 1 else '0'
#             max_span_labels.append(max_span_label)

#             rel_toks.append('_'.join([str(k), str(i), str(i)]))
#             lca_labels.append(label)

#         for j in range(i+1,len(sent)):
#             j_tok = sent[j]
#             if punct_regex.match(j_tok):
#                 continue
#                 # default case: no traces
#             lowest_common_ancestor = tree_notr.treeposition_spanning_leaves(i,j+1)
#             label = find_label(tree_notr, lowest_common_ancestor).split('-')[0]
#                 # outfile_only_pos.write('\t'.join([str(i_preproc), str(j_preproc), label]) + '\n') # , preproc_sent[i_preproc], preproc_sent[j_preproc]])+'\n')

#             lca_node = find_node(tree_notr, lowest_common_ancestor)
#             max_span_label = '0'
#             if len(lca_node.leaves()) == 1 + j - i:
#                 max_span_label = '1'
#             else:
#                 i_tok_matches = lca_node.leaves()[0] == i_tok or (punct_regex.match(lca_node.leaves()[0]) and lca_node.leaves()[1] == i_tok)
#                 j_tok_matches = lca_node.leaves()[-1] == j_tok or (punct_regex.match(lca_node.leaves()[-1]) and lca_node.leaves()[-2] == j_tok)
#                 if i_tok_matches and j_tok_matches:
#                     max_span_label = '1'
            
            
#             if not(specialCondition=='NP_embed'):
#                 lca_labels.append(label)
#                 rel_toks.append('_'.join([str(k), str(i), str(j_preproc)]))
#                 max_span_labels.append(max_span_label)
#             elif label=='NP' and specialConditionNPEmbed(tree_notr, i, j, emb_depth=emb_depth):
#                 lca_labels.append(label)
#                 rel_toks.append('_'.join([str(k), str(i), str(j_preproc)]))
#                 max_span_labels.append(max_span_label)

#     return rel_toks,lca_labels,max_span_labels

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

    ###############
    # PREP
    ###############
    # model=$1
    # traincutoff=$2
    # testcutoff=$3
    # layersel="second"
    # datadir="data/"
    # modeldir=$datadir"/"$model"/"
    # mkdir $datadir
    # mkdir $modeldir

    """
    Call with: python extract_labels.py -data pcfg-lm/src/lm_training/corpora/eval_trees_10k.txt 
    -text_toks data/train_text.txt -rel_toks data/train_rel_toks.txt -rel_labels data/train_rel_labels.txt 
    -next -shared_levels data/train_shared_levels.txt -unary data/train_unaries.txt -cutoff 2 -max_sent_length 31
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-data')
    parser.add_argument('-text_toks') 
    parser.add_argument('-rel_toks')
    parser.add_argument('-rel_labels') 
    parser.add_argument('-max_span_const')
    parser.add_argument('-cutoff')
    parser.add_argument('-max_sent_length')
    parser.add_argument('-np_embed', action='store_true', default=False)
    parser.add_argument('-next', action='store_true', default=False) # run in experiments
    parser.add_argument('-shared_levels', default=None) # run in experiments
    parser.add_argument('-unary', default=None) # run in experiments
    parser.add_argument('-tree_out', default=None, help="optional file where all trees are printed that are processed (and not skipped)")
    parsedargs = parser.parse_args()

    ignored_sents = [] # added if not length: 3 < length < 31
    ignore_list = [] # no idea but this is in the original code

    # Set thresholds
    cutoff = np.inf
    if parsedargs.cutoff is not None:
        cutoff = int(parsedargs.cutoff)
    max_sent_length = np.inf
    if parsedargs.max_sent_length is not None:
        max_sent_length = int(parsedargs.max_sent_length)

    # Reading in input trees
    with open(parsedargs.data) as f:
        tree_corpus = [nltk.Tree.fromstring(l.strip()) for l in f]

    # Open output files
    text_toks_file = open(parsedargs.text_toks, 'w')
    rel_toks_file = open(parsedargs.rel_toks, 'w')
    rel_labels_file = open(parsedargs.rel_labels, 'w')

    if parsedargs.max_span_const is not None:
        max_span_file = open(parsedargs.max_span_const, 'w')
    if parsedargs.shared_levels is not None:
        shared_levels_file = open(parsedargs.shared_levels, 'w')
    if parsedargs.unary is not None:
        unary_file = open(parsedargs.unary, 'w')
    if parsedargs.tree_out is not None:
        tree_out_file = open(parsedargs.tree_out, 'w') 
    binary = False

    specialCond = 'NP_embed' if parsedargs.np_embed else False 
    if specialCond:
        emb_depth=int(parsedargs.np_embed)
    else:
        emb_depth=None
    
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

            rel_toks, lca_labels, shared_levels_rel, unary_labels = phrasePropertiesForAdjacentTokenPairsInPTB(k - len(ignored_sents), tree)
            assert len(rel_toks) == len(lca_labels) == len(shared_levels_rel) == len(unary_labels)
        # else:
            # # not yet edited
            # rel_toks, lca_labels, max_span_labels = phrasePropertiesForTokenPairsInPTB(k - len(ignored_sents), tree.leaves(), tree, specialCondition=specialCond, emb_depth=emb_depth)
            # assert len(rel_toks) == len(lca_labels) == len(max_span_labels)

        rel_toks_file.write(' '.join(rel_toks) + '\n')
        rel_labels_file.write(' '.join(lca_labels) + '\n')
        text_toks_file.write(' '.join(tree.leaves()) + '\n')

        if parsedargs.shared_levels is not None:
            shared_levels_file.write(' '.join(shared_levels_rel) + '\n')
        if parsedargs.unary is not None:
            unary_file.write(' '.join(unary_labels) + '\n')
        if parsedargs.max_span_const is not None:
            max_span_file.write(' '.join(max_span_labels) + '\n')
        if parsedargs.tree_out is not None:
            if len(tree_notr) == 1 and tree_notr.label() == 'VROOT':
                tree_notr = tree_notr[0]
            tree_out_file.write(str(tree_notr).replace('\n','').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ')+'\n')

    text_toks_file.close()
    rel_toks_file.close()
    rel_labels_file.close()
    if parsedargs.max_span_const is not None:
        max_span_file.close()
    if parsedargs.shared_levels is not None:
        shared_levels_file.close()
    if parsedargs.unary is not None:
        unary_file.close()
    if parsedargs.tree_out is not None:
        tree_out_file.close()

    print('finished. ignored sentences: ', ignored_sents)
