import tempfile
import subprocess
import warnings
import os
import codecs

import tempfile
import copy
import sys
import time
from tree import SeqTree, RelativeLevelTreeEncoder
from logging import warning
from pathlib import Path
import nltk
import pickle

def incorporate_unaries(sentences, output, join_char="~", split_char="@"):
    sentence = []
    preds = []

    for ((word,postag), pred) in zip(sentences[1:-1], output[1:-1]):
        # loop through words in sentence
        if pred.split(split_char)[2] != 'XX':
            # word is an unary leaf
            sentence.append((word, pred.split(split_char)[2] + join_char + postag))                     
        else:
            # word is part of multileave constituent
            sentence.append((word,postag)) 

        preds.append(pred)
    
    return sentence, preds

def sequence_to_parenthesis(sentences, labels, join_char="~", split_char="@"):
    """
    Transforms a list of sentences and predictions (labels) into parenthesized trees
    @param sentences: A list of list of (word,postag)
    @param labels: A list of list of predictions
    @return A list of parenthesized trees
    """
    parenthesized_trees = []  
    relative_encoder = RelativeLevelTreeEncoder(join_char=join_char, split_char=split_char)
    
    f_max_in_common = SeqTree.maxincommon_to_tree
    f_uncollapse = relative_encoder.uncollapse
    
    total_posprocessing_time = 0
    for n, output in enumerate(labels):       
        # loop through sentences
        sentence, preds = incorporate_unaries(sentences[n], output)
        tree = f_max_in_common(preds, sentence, relative_encoder)
        print(tree.pretty_print())          
        # Removing empty label from root
        if tree.label() == SeqTree.EMPTY_LABEL:
            #If a node has more than two children
            #it means that the constituent should have been filled.
            if len(tree) > 1:
                print ("WARNING: ROOT empty node with more than one child")
            else:
                while (tree.label() == SeqTree.EMPTY_LABEL) and len(tree) == 1:
                    tree = tree[0]

        #Uncollapsing the root. Rare needed
        if join_char in tree.label():
            aux = SeqTree(tree.label().split(join_char)[0],[])
            aux.append(SeqTree(join_char.join(tree.label().split(join_char)[1:]), tree ))
            tree = aux

        tree = f_uncollapse(tree)

        # total_posprocessing_time+= time.time()-init_parenthesized_time
        # #To avoid problems when dumping the parenthesized tree to a file
        # aux = tree.pformat(margin=100000000)
        
        # if aux.startswith("( ("): #Ad-hoc workarounf for sentences of length 1 in German SPRML
        #     aux = aux[2:-1]
        tree.pretty_print()
        parenthesized_trees.append(tree)

    return parenthesized_trees 


def get_postag_trees(tree):
    """
    Gets a list of the PoS tags from the tree
    @return A list containing (word, postag)
    """
    postags = []
    
    for _, child in enumerate(tree):
        if len(child) == 1 and type(child[-1]) == type(""):
            word = child.leaves()[0]
            label = child.label().split("_")[0]
            postags.append((word, label))
        else:
            postags.extend(get_postag_trees(child))
    
    return postags

if __name__ == '__main__':
    home_path = Path("/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code")
    labels = []

    preds = []
    with open(home_path / Path('data/combined_predictions.txt')) as f:
        # combined predictions in the form of LEVEL@LCA_LABEL@UNARY
        for line in f:
            preds.append(line.strip('\n').split(' '))

    sentences = []
    with open(home_path / Path('data/sentences_postags.pickle'), 'rb') as f:
        sentences = pickle.load(f)
    print(preds[0])
    print(sentences[0])
    sequence_to_parenthesis([sentences[0]], [preds[0]])

    with open(home_path / Path('corpora/eval_trees_10k.txt')) as f:
        tree_corpus = [nltk.Tree.fromstring(l.strip()) for l in f]
    # tree_corpus[0].pretty_print()




                