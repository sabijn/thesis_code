from nltk import PCFG as nltk_PCFG, Production, ProbabilisticProduction
from nltk.grammar import Nonterminal
from nltk.parse import IncrementalLeftCornerChartParser as Parser

from classes import (TokenizerConfig, 
                        Tokenizer, 
                        PCFGConfig,
                        PCFG)

from collections import defaultdict
from typing import Dict
from tqdm import tqdm
import numpy as np
import signal
import time
import pickle
import torch
import argparse

REPLACEMENTS = [
    ('``', 'TICK'),
    ('.', 'DOT'),
    ('#', 'HASH'),
    ('-LRB-', 'LRB'),
    ('-RRB-', 'RRB'),
    ("''", 'APOSTROPHE'),
    ('$', 'DOLLAR'),
    (':', 'COLON'),
    (',', 'COMMA'),
    ('@', 'AT'),
    ('PRT|ADVP', 'PRTADVP'),
    ('^g', ''),
]

def format_line(line):
    for s1, s2 in REPLACEMENTS:
        line = line.replace(s1, s2)
    
    return line

INT2NT = ['ROOT', 'S^g', '@S^g', 'NP^g', 'NNP', 'VP^g', 'VBD', ',', 'CC', 'DT', 'NN', '.', 'PRP', 'MD', '@VP^g', 'VB', 'ADVP^g', 'RB', 'PP^g', 'IN', 'NNS', ':', 'VBN', 'INTJ^g', 'UH', 'ADJP^g', '@ADJP^g', 'JJ', 'SBAR^g', 'EX', '@NP^g', 'TO', 'WHNP^g', 'WP', 'POS', 'PRP$', 'JJR', 'VBG', 'RRC^g', 'JJS', 'CD', 'PRT^g', 'RP', '@PP^g', '@ADVP^g', '``', 'VBP', 'VBZ', "''", 'WHADVP^g', 'WRB', 'WDT', 'FRAG^g', 'PDT', '@SBAR^g', 'RBR', 'QP^g', '@QP^g', 'NNPS', 'RBS', 'NX^g', '@NX^g', 'PRN^g', '@PRN^g', '@FRAG^g', '@INTJ^g', 'CONJP^g', '@CONJP^g', 'SQ^g', '@SQ^g', 'SBARQ^g', '@SBARQ^g', 'SINV^g', '@SINV^g', 'UCP^g', '@UCP^g', '@WHNP^g', 'WHADJP^g', 'X^g', 'SYM', 'FW', 'WP$', 'WHPP^g', '-LRB-', '-RRB-', '$', '@PRT^g', 'NAC^g', '@NAC^g', '@WHADJP^g', 'LS', '@WHADVP^g', 'LST^g', '@X^g', '@LST^g', '#', '@RRC^g']
INT2NT = [format_line(nt) for nt in INT2NT]

def load_language(args, encoder="transformer", corpus_size=200_000):
    if args.hardware == 'snellius':
        grammar_file = f'/scratch-shared/sabijn/{args.version}/subset_pcfg_{args.top_k}.txt'
        corpus_file = f'/scratch-shared/sabijn/{args.version}/corpus_{args.top_k}_{args.version}.pt'
    
    elif args.hardware == 'local':
        grammar_file = f'grammars/nltk/{args.version}/subset_pcfg_{args.top_k}.txt'
        corpus_file = f'corpora/{args.version}/corpus_{args.top_k}_{args.version}.pt'
    
    else:
        raise NotImplementedError(args.hardware)


    tokenizer_config = TokenizerConfig(
            add_cls=(encoder == "transformer"),
            masked_lm=(encoder == "transformer"),
            unk_threshold=5,
        )
    
    tokenizer = Tokenizer(tokenizer_config)

    config = PCFGConfig(
        is_binary=False,
        min_length=6,
        max_length=25,
        max_depth=25,
        corpus_size=args.corpus_size,
        grammar_file=grammar_file,
        start="S_0",
        masked_lm=(encoder == "transformer"),
        allow_duplicates=True,
        split_ratio=(0.8,0.1,0.1),
        use_unk_pos_tags=True,
        verbose=True,
        store_trees=True,
        output_dir='.',
        top_k=args.top_k,
        version=args.version,
        file=corpus_file
    )

    language = PCFG(config, tokenizer)

    return language, tokenizer


def add_special_token(grammar):
    leaf_prod_lhs = set(prod.lhs() for prod in grammar.productions() if isinstance(prod.rhs()[0], str))

    special_token = '<X>'
    special_prods = []

    # Add single '<X>' leaf that is added to all leaf_prods and parse once
    for lhs in leaf_prod_lhs:
        special_prod = ProbabilisticProduction(lhs, (special_token,), prob=1.)
        special_prods.append(special_prod)

    grammar._productions.extend(special_prods)

    grammar._calculate_indexes()
    grammar._calculate_grammar_forms()
    grammar._calculate_leftcorners()


def create_prod2prob_dict(grammar) -> Dict[Production, float]:
    prod2prob = defaultdict(float)

    for lhs, prods in grammar._lhs_index.items():
        for prod in prods:
            cfg_prod = Production(lhs, prod.rhs())

            prod2prob[cfg_prod] = prod.prob()
            
    return prod2prob

class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException

signal.signal(signal.SIGALRM, timeout_handler)

def get_subset(args, lm_language):
    file = open(f'perplexities/babyberta/optimal_ppl_mlm_normal_{args.top_k}_size10000.pkl', 'rb')

    # dump information to that file
    data = pickle.load(file)
    file.close()

    _, _, _, _, sen_ids = data
    corpus = lm_language.test_corpus
    subset_corpus = [corpus[sen_id] for sen_id in sen_ids]
    
    with open(f'corpora/{args.version}/test_sent_normal_0.8_subset.txt', 'w') as f:
        # write to txt file
        for sen in subset_corpus:
            f.write(sen + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate the optimal perplexity')
    parser.add_argument('--top_k', type=float, default=0.2)
    parser.add_argument('--version', type=str, default='normal')
    parser.add_argument('--output_file', type=str, default='perplexities')
    parser.add_argument('--corpus_size', type=int, default=None) 
    parser.add_argument('--parse_method', type=str, default='all_parses', choices= ['all_parses', 'sen_parses', 'current_parse']) 
    parser.add_argument('--max_parse_time', type=int, default=10)
    parser.add_argument('--hardware', type=str, default='local', choices=['snellius', 'local'])
    parser.add_argument('--model', type=str, default='babyberta', choices=['gpt2', 'babyberta'])

    args = parser.parse_args()

    language, tokenizer = load_language(args, encoder="transformer", corpus_size=args.corpus_size)
    add_special_token(language.grammar)
    prod2prob = create_prod2prob_dict(language.grammar)

    get_subset(args, language)

    del language
    del tokenizer