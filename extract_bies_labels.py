from treetoolbox import find_node, load_ptb, np_regex, vp_regex, tr_leaf_regex, find_end_indices, find_xps_above_i, address_is_xp, find_label, find_tracing_alignment, find_trace_ix, preprocess, lowest_phrase_above_leaf_i
# during test: from data_prep.treetoolbox
import sys
import argparse
import re
import numpy as np
import nltk
from tqdm import tqdm

punct_regex = re.compile(r"[^\w][^\w]?")

def findBIESTag(i: int, tree : nltk.Tree, i_tok: str, with_phrase_label=False) -> str:
    """
    Find BIES tag for token i in tree. If multiple tags, than the one
    with the lowest phrase label is chosen.
    Input:
        i: index of token
        tree: nltk.Tree
        i_tok: token at index i
    Output:
        tag: BIES tag for token i
    """
    # Check if token is punctuation
    if punct_regex.match(i_tok):
        return 'PCT'
    
    phrase_label, phrase_node, ga_of_phrase_node = lowest_phrase_above_leaf_i(i, tree, return_target_ga=True)

    ga_of_leaf = tree.treeposition_spanning_leaves(i,i+1)
    ga_phrase_to_leaf = ga_of_leaf[len(ga_of_phrase_node):]
    is_beginning = ga_phrase_to_leaf[0] == 0 and (len(set(ga_phrase_to_leaf))==1)
    is_end = True
    node = phrase_node

    for k in ga_phrase_to_leaf:
        if len(node) - 1 > k:
            is_end=False
            break
        else:
            node = node[k]

    # Assign shortest BIES tag   
    if is_beginning and is_end:
        # Single-token phrase
        tag = 'S'
    elif is_beginning:
        # Beginning of phrase
        tag = 'B'
    elif is_end:
        # End of phrase
        tag = 'E'
    else:
        # Inside phrase
        tag = 'I'

    if with_phrase_label:
        if phrase_label.startswith('NP') and len(phrase_label.split('-')) > 1:
            tag+='-'+'-'.join(phrase_label.split('-')[:2])
        else:
            tag+='-'+phrase_label.split('-')[0]
    return tag

def biesLabels(tree, with_phrase_labels=False):
    """
    Loop through through leaves in tree and assign BIES labels to each token
    Input:
        tree: nltk.Tree
    Output: 
        text_toks: list of tokens in tree
        bies_labels: list of BIES labels for each token in tree
    """
    sent = tree.leaves()
    text_toks = []
    bies_labels = []

    for i in range(len(sent)):
        i_tok = sent[i]

        # find phrase above token i
        label = findBIESTag(i, tree, i_tok, with_phrase_label=with_phrase_labels)

        text_toks.append(i_tok)
        bies_labels.append(label)

    return text_toks, bies_labels

if __name__=='__main__':
    """
    Usage: 
    python3 span_prediction_format -ptb_tr <file_pattern> -ptb_notr <file_pattern> -text_toks <filename> -bies_labels <filename>

    Input Options: 
    - PTB with and without traces (both are needed)

    Output Options (each file has same number of lines): 
    - text_toks txt file with one sentence per line (input file for computing activations)
    - bies_labels txt file where the k-th label in the l-th line represents the beginning/inside/end/only label in the PTB of the corresponding tokens in text_toks.txt

    other options
    - cutoff x an integer x such that the scripts stops after processing x sentences
    -max_sent_length an integer x for the maximum sentence length (suggested: 20)
    
    The script asserts that all output files have the same number of lines, and that the output files text_toks and bies_labels have the same number of elements per line
    """

    ###############
    # PREP
    ###############

    """
    Call with: python extract_bies_labels.py -data pcfg-lm/src/lm_training/corpora/eval_trees_10k.txt 
    -text_toks data/train_text_bies.txt -bies_labels data/train_bies_labels.txt -max_sent_length 31
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-data')
    parser.add_argument('-text_toks') 
    parser.add_argument('-bies_labels') 
    parser.add_argument('-cutoff')
    parser.add_argument('-max_sent_length')
    parser.add_argument('-with_phrase_labels', action='store_true')
    parsedargs = parser.parse_args()

    ignored_sents = []
    ignore_list = []

    # Probably used during development with a lower cutoff
    cutoff = np.inf
    if parsedargs.cutoff is not None:
        cutoff = int(parsedargs.cutoff)
    max_sent_length = np.inf
    if parsedargs.max_sent_length is not None:
        max_sent_length = int(parsedargs.max_sent_length)
    
        # Reading in input trees
    with open(parsedargs.data) as f:
        tree_corpus = [nltk.Tree.fromstring(l.strip()) for l in f]

    text_toks_file = open(parsedargs.text_toks, 'w')
    bies_labels_file = open(parsedargs.bies_labels, 'w')

    binary = False
    label_counts = dict()

    output_sents = set()
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

        with_phrase_labels = parsedargs.with_phrase_labels
        preproc_sent, bies_labels = biesLabels(tree, with_phrase_labels=with_phrase_labels)

        assert len(preproc_sent) == len(bies_labels)

        # Do not store the same sentence twice!
        if str(preproc_sent) in output_sents:
            print('Found duplicate sentence, skipping: ', preproc_sent)
            continue
        output_sents.add(str(preproc_sent))

        for l in bies_labels:
            if l not in label_counts:
                label_counts[l] = 0
            label_counts[l] = label_counts[l] + 1

        bies_labels_file.write(' '.join(bies_labels) + '\n')
        text_toks_file.write(' '.join(preproc_sent) + '\n')

    text_toks_file.close()
    bies_labels_file.close()

    print('Finished. ignored sentences: ', ignored_sents)
    print('Label distribution: ')
    label_counts = dict(sorted(label_counts.items(), key=lambda item: item[1], reverse=True))
    total = sum(label_counts.values())
    print(label_counts)
    for l,c in label_counts.items():
        print(l,'\t', c, '\t', str(float(c/total)))