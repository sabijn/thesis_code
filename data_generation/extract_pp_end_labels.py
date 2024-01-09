from treetoolbox import lowest_phrase_above_leaf_i
# during test: from data_prep.treetoolbox
import sys
import argparse
import re
import numpy as np
import nltk
from tqdm import tqdm
from main import load_model
import torch
from pathlib import Path

punct_regex = re.compile(r"[^\w][^\w]?")

def find_node(tree, address):
    """return subtree at address"""

    subtree = tree
    for i in address:
        subtree = subtree[i]
    return subtree

def find_pp(i, tree, sent, labels: dict):
    ga_of_target = tree.treeposition_spanning_leaves(i,i+1)[:-2]
    node = find_node(tree, ga_of_target)
    label = node.label()

    label = re.sub('[^A-Za-z]+', '', label)
    if label == 'PP':
        try:
            ending_index = sent.index(node.leaves()[-1])
            labels[ending_index] = 1
        except:
            return labels

    return labels

def biesLabels(tree, tokenizer, with_phrase_labels=False, skip_unkown_tokens=True):
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
    
    for i in range(len(sent)):
        i_tok = sent[i]
        # find phrase above token i
        if skip_unkown_tokens:
            if i_tok not in tokenizer.vocab:
                continue
        text_toks.append(i_tok)

    pp_labels = np.zeros(len(text_toks))
    for i in range(len(text_toks)):
        i_tok = text_toks[i]
        
        pp_labels = find_pp(i, tree, text_toks, pp_labels)

    return text_toks, pp_labels

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
    Call with: python extract_pp_end_labels.py -data pcfg-lm/src/lm_training/corpora/eval_trees_10k.txt -text_toks data/train_text_bies.txt -bies_labels data/train_pp_labels.txt -max_sent_length 31
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
    
    model_path = Path('pcfg-lm/resources/checkpoints/deberta/')
    model, tokenizer = load_model(model_path, device)

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
        preproc_sent, bies_labels = biesLabels(tree, tokenizer, with_phrase_labels=with_phrase_labels, skip_unkown_tokens=True)
        # tree.pretty_print()
        # print(bies_labels)

        assert len(preproc_sent) == bies_labels.shape[0], f'Number of tokens ({len(preproc_sent)}) and number of labels ({bies_labels.shape[0]}) do not match.'

        # Do not store the same sentence twice!
        if str(preproc_sent) in output_sents:
            print('Found duplicate sentence, skipping: ', preproc_sent)
            continue
        output_sents.add(str(preproc_sent))

        bies_labels_file.write(np.array2string(bies_labels) + '\n')
        text_toks_file.write(' '.join(preproc_sent) + '\n')

    text_toks_file.close()
    bies_labels_file.close()

    # print('Finished. ignored sentences: ', ignored_sents)
    # print('Label distribution: ')
    # label_counts = dict(sorted(label_counts.items(), key=lambda item: item[1], reverse=True))
    # total = sum(label_counts.values())
    # print(label_counts)
    # for l,c in label_counts.items():
    #     print(l,'\t', c, '\t', str(float(c/total)))