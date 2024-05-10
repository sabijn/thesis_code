from nltk import PCFG as nltk_PCFG, Production, ProbabilisticProduction
from nltk.grammar import Nonterminal
from nltk.parse import IncrementalLeftCornerChartParser as Parser

from classes import (TokenizerConfig, 
                        Tokenizer, 
                        PCFGConfig,
                        PCFG)

from collections import defaultdict
from typing import Dict
from tqdm import tqdm_notebook
import numpy as np
import signal
import time
import pickle
import torch

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

def load_language(top_k, version, encoder="transformer", corpus_size=200_000):
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
        corpus_size=1_000_000,
        grammar_file=f'grammars/nltk/{version}/subset_pcfg_{top_k}.txt',
        start="S_0",
        masked_lm=(encoder == "transformer"),
        allow_duplicates=True,
        split_ratio=(0.8,0.1,0.1),
        use_unk_pos_tags=True,
        verbose=True,
        store_trees=True,
        output_dir='.',
        top_k=top_k,
        version=version,
        file=f'corpora/{version}/corpus_{top_k}_{version}.pt'
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


def pcfg_perplexity(lm_language, method, prod2prob, max_parse_time=10, corpus_size=None, sen_ids_filter=None, verbose=False):
    all_probs = []
    sen_lens = []
    num_parses = []
    sen_ids = []
    
    chart_parser = Parser(lm_language.grammar)
    corpus = lm_language.test_corpus[:corpus_size]  
    iterator = tqdm_notebook(corpus) if verbose else corpus
    
    for sen_idx, sen in enumerate(iterator):
        if sen_ids_filter is not None and sen_idx not in sen_ids_filter:
            continue

        orig_tree = lm_language.tree_corpus[sen]
        sen = sen.split()
        sen_len = len(sen)

        if method == 'all_parses':
            weighted_leaf_probs = []
            num_sen_parses = []
            skip = False

            signal.alarm(max_parse_time)
            try:
                for idx, orig_leaf in enumerate(sen):
                    sen2 = list(sen)
                    sen2[idx] = '<X>'

                    tree_probs = []
                    leaf_probs = []

                    for i, tree in enumerate(chart_parser.parse(sen2)):
                        tree_prob = np.prod([(prod2prob[prod]) for prod in tree.productions()])

                        leaf_idx_prod = [prod for prod in tree.productions() if isinstance(prod.rhs()[0], str)][idx]
                        leaf_idx_pos = leaf_idx_prod.lhs()
                        orig_leaf_prob = prod2prob[Production(leaf_idx_pos, (orig_leaf,))]

                        tree_probs.append(tree_prob)
                        leaf_probs.append(orig_leaf_prob)

                    num_sen_parses.append(i+1)
                    tree_probs_sum = np.sum(tree_probs)

                    weighted_leaf_prob = sum((tree_prob/tree_probs_sum) * leaf_prob for tree_prob, leaf_prob in zip(tree_probs, leaf_probs))
                    weighted_leaf_probs.append(np.log(weighted_leaf_prob))
            except TimeoutException:
                continue
            finally:
                signal.alarm(0)

            sen_ids.append(sen_idx)
            num_parses.append(num_sen_parses)
            print(np.sum(weighted_leaf_probs))
            all_probs.append(np.sum(weighted_leaf_probs))  
                  
        elif method == 'sen_parses':
            sen_leaf_probs = []
            sen_tree_probs = []

            start_time = time.time()
            signal.alarm(max_parse_time)
            try:
                parses = list(chart_parser.parse(sen))
            except TimeoutException:
                continue
            finally:
                signal.alarm(0)

            for i, tree in enumerate(parses):
                leaf_probs = [prod2prob[prod] for prod in tree.productions() if isinstance(prod.rhs()[0], str)]
                leaf_prob = np.prod(leaf_probs)
                tree_prob = np.prod([(prod2prob[prod]) for prod in tree.productions()])# if not isinstance(prod.rhs()[0], str)])

                sen_leaf_probs.append(leaf_prob)
                sen_tree_probs.append(tree_prob)

            total_sen_tree_probs = sum(sen_tree_probs)
            weighted_sen_prob = sum(
                (tree_prob/total_sen_tree_probs) * leaf_prob 
                for tree_prob, leaf_prob in zip(sen_tree_probs, sen_leaf_probs)
            )
            weighted_sen_logprob = np.log(weighted_sen_prob)

            sen_ids.append(sen_idx)
            num_parses.append(i+1)
            all_probs.append(weighted_sen_logprob)
        elif method == 'current_parse':
            leaf_prods = [prod for prod in orig_tree.productions() if isinstance(prod.rhs()[0], str)]
            sen_prob = np.sum([np.log(prod2prob[prod]) for prod in leaf_prods])
            
            sen_ids.append(sen_idx)
            all_probs.append(sen_prob)
        else:
            raise ValueError(method)

        sen_lens.append(sen_len)
                
    avg_ppl = np.exp(-np.sum(all_probs)/np.sum(sen_lens))
    
    return avg_ppl, all_probs, num_parses, sen_lens, sen_ids

if __name__ == '__main__':
    top_k = 0.2
    VERSION = 'normal'
    output = {}
    corpus_size = 10
    output_file = f'perplexities/optimal_ppls_{VERSION}.pkl'


    language, tokenizer = load_language(top_k, VERSION, encoder="transformer", corpus_size=corpus_size)
    add_special_token(language.grammar)
    prod2prob = create_prod2prob_dict(language.grammar)

    avg_ppl, all_probs, num_parses, sen_lens, sen_ids = pcfg_perplexity(
        language, 'current_parse', prod2prob, max_parse_time=1, corpus_size=corpus_size, 
    )
    print('probs', all_probs)
    print('ids', sen_ids)
    print('lens', sen_lens)
    print('All probs: ', len(all_probs))

    output[top_k] = avg_ppl
    print(f'The optimal perplexity of {VERSION} {top_k}: {avg_ppl}')
    assert len(all_probs) == len(language.test_corpus)

    del language
    del tokenizer