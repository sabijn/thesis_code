import nltk
from nltk import ProbabilisticProduction, PCFG, Nonterminal
from tqdm import tqdm
import random
from typing import List, Tuple, Dict, Any
import itertools
from collections import Counter, defaultdict
import pickle
import os
from nltk import PCFG as nltk_PCFG
import argparse

def sort_and_select_productions(rhs_prods, top_k):
    """
    Sort rhs probabilities and select top_k productions.
    """
    sorted_rhs = sorted(rhs_prods, key=lambda prod: -prod.prob())
    
    # Remove recursive productions
    # TODO: ask Jaap if here should be a recursion flag?
    sorted_rhs = [prod for prod in sorted_rhs if (prod.lhs() not in prod.rhs())]
    
    # Select top_k productions
    if isinstance(top_k, int):
        subset_rhs = sorted_rhs[:top_k]

    # Select top_k productions based on probability
    elif isinstance(top_k, float):
        acc_prob = 0.
        subset_rhs = []
        for prod in sorted_rhs:
            subset_rhs.append(prod)
            acc_prob += prod.prob()
            if acc_prob > top_k:
                break
    
    return subset_rhs


def group_productions_by_lhs(productions):
    """
    Group productions by their left-hand side symbol.
    """
    lhs_productions = defaultdict(list)
    for prod in productions:
        lhs_productions[prod.lhs()].append(prod)

    return lhs_productions


def create_subset_productions(productions, top_k, lexical=False):
    """
    Given a list of productions, create a subset of productions by selecting the top_k most probable productions.
    """
    subset_productions = []

    # Group productions by their left-hand side
    prob_productions_dict = group_productions_by_lhs(productions)

    # Sort productions by probability (from high to low) and select top_k (for each left-hand side symbol)
    for rhs_prods in prob_productions_dict.values():
        if lexical:
            # Separate terminal and nonterminal productions
            terminal_prods = [prod for prod in rhs_prods if all(isinstance(sym, str) for sym in prod.rhs())]
            nonterminal_prods = [prod for prod in rhs_prods if not all(isinstance(sym, str) for sym in prod.rhs())]
            non_terminal_subset = sort_and_select_productions(nonterminal_prods, top_k)
            subset_rhs = terminal_prods + non_terminal_subset
        else:
            subset_rhs = sort_and_select_productions(rhs_prods, top_k)

        subset_productions.extend(subset_rhs)

    return subset_productions



def reachable_productions(productions, lhs, parents=tuple(), prods_seen=set(), no_recursion=False):
    """
    Create a generator that yields all reachable productions from a given lhs symbol.
    """
    # reminder: *(tuple) unpacks the tuple into arguments
    new_parents = (*parents, lhs)
    
    # select productions belonging to the current lhs
    lhs_productions = [prod for prod in productions if prod.lhs() == lhs]
    
    for prod in lhs_productions:
        # reminder: creates a tuple with one element (unmutable)
        if (prod,) in prods_seen:
            continue
        prods_seen.add((prod,))

        # check if the rhs contains a parent symbol    
        if no_recursion and any([rhs in parents for rhs in prod.rhs()]):
            continue

        yield prod

        for rhs in prod.rhs():
            if isinstance(rhs, Nonterminal):
                yield from reachable_productions(
                    productions, 
                    rhs, 
                    parents=new_parents,
                    prods_seen=prods_seen,
                    no_recursion=no_recursion,
                )


def is_leaf(prod):
    """
    Check if a production is a leaf (i.e. has only one rhs symbol which is a string)
    """
    return len(prod.rhs()) == 1 and isinstance(prod.rhs()[0], str)


def leaves_to_pos(prods):
    """
    Labels each leaf with its POS tag (with a probability of 1.0)
    """
    return set(ProbabilisticProduction(prod.lhs(), 
                                       (prod.lhs().symbol().lower(),), 
                                       prob=1.0) if is_leaf(prod) else prod for prod in prods)

def renormalize_probs(prods):
    """
    Renomalize probabilities of productions for each left-hand side symbol.
    """
    new_prods = []
    all_lhs = set(prod.lhs() for prod in prods)
    
    for lhs in all_lhs:
        lhs_prods = [prod for prod in prods if prod.lhs() == lhs]
        
        lhs_total_prob = sum(prod.prob() for prod in lhs_prods)
        
        for prod in lhs_prods:
            new_prob = prod.prob() / lhs_total_prob
            new_prods.append(
                ProbabilisticProduction(prod.lhs(), prod.rhs(), prob=new_prob)
            )
            
    return new_prods


def create_lookup_probs(pcfg):
    """
    Create probability lookup table
    """
    pcfg._lhs_prob_index = {}
    for lhs in pcfg._lhs_index.keys():
        lhs_probs = [prod.prob() for prod in pcfg.productions(lhs=lhs)]
        pcfg._lhs_prob_index[lhs] = lhs_probs
    
    return pcfg

def write_to_txt(pcfg, filename) -> None:
    f = open(filename, 'w')
    for prod in pcfg.productions():
        lhs = prod.lhs()
        # Check each element in rhs; if it's a string, add quotations
        rhs_with_quotes = []
        for item in prod.rhs():
            if isinstance(item, str):  # Assuming terminals are represented as strings
                rhs_with_quotes.append(f"\'{item}\'")
            else:
                rhs_with_quotes.append(str(item))
        rhs_formatted = ' '.join(rhs_with_quotes)
        prob = prod.prob()
        # Format the probability as a decimal float with desired prec
        # ision, e.g., 10 decimal places
        formatted_prob = f'{prob:.10f}'
        # Recreate the production string with the formatted probability and rhs with quotations
        prod_str = f"{lhs} -> {rhs_formatted} [{formatted_prob}]"
        f.write(f"{prod_str}\n")
    f.close()


def create_subset_pcfg(productions, args, top_k=0.2, no_recursion=False, save=True, lexical=False):
    """
    Create a subset PCFG from the original PCFG by selecting the top_k most probable productions.
    """
    start = Nonterminal('S_0')

    print(f'************ Creating subset PCFG with top k = {top_k}... ************', flush=True)
    print(f'Starting with {len(productions)} productions.', flush=True)
    subset_productions = create_subset_productions(productions, top_k, lexical=lexical)
    print(f'Created subset PCFG with a length of {len(subset_productions)} productions.', flush=True)

    print('Cleaning subset: (1) removing unreachable productions...')
    final_subset_productions = set(
        reachable_productions(
            subset_productions, 
            start, 
            no_recursion=no_recursion,
        )
    )

    # update set for removed recursive productions
    reachable_nonterminals = set(prod.lhs() for prod in final_subset_productions)
    print('Amount of reachable nonterminals:', len(reachable_nonterminals))
    final_subset_productions = [
        prod for prod in final_subset_productions 
        if all([rhs in reachable_nonterminals for rhs in prod.rhs()]) or is_leaf(prod)
    ]
    print(f'Finished cleaning subset (1) left with {len(final_subset_productions)} productions.')

    print('Cleaning subset: (2) renormalizing probabilities...')
    final_subset_productions = renormalize_probs(final_subset_productions)
    print(f'Finished cleaning subset (2)')

    print('Cleaning subset: (3) adding POS tags...')
    pos_productions = leaves_to_pos(final_subset_productions)
    pos_productions = renormalize_probs(pos_productions)
    print('Finished cleaning subset (3)')

    # subset_pcfg does not contain pos_tags
    subset_pcfg = PCFG(start, final_subset_productions)
    subset_pcfg_pos = PCFG(start, pos_productions)
    

    print('Write subset PCFG to pickle...')
    write_to_txt(subset_pcfg, f'{args.output_dir}/subset_pcfg_{top_k}.txt')
    write_to_txt(subset_pcfg_pos, f'{args.output_dir}/subset_pcfg_{top_k}_pos.txt')

    print('Done')
    
    return subset_pcfg, subset_pcfg_pos

def load_subset_pcfg(prob_productions, args, top_k=0.2, save=True, load=True, lexical=False, no_recursion=False):
    filename = f'grammars/nltk/subset_pcfg_{top_k}.pkl'
    filename_pos = f'grammars/nltk/subset_pcfg_{top_k}_pos.pkl'
    
    if load:
        if os.path.exists(filename) and os.path.exists(filename_pos):
            with open(filename, 'rb') as f:
                subset_pcfg = pickle.load(f)
            
            with open(filename_pos, 'rb') as f:
                subset_pcfg_pos = pickle.load(f)

            return subset_pcfg, subset_pcfg_pos
    
    subset_pcfg, subset_pcfg_pos = create_subset_pcfg(prob_productions, args, top_k, save=save, lexical=lexical, no_recursion=no_recursion)

    return subset_pcfg, subset_pcfg_pos

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pcfg_dir', type=str, default='grammars/nltk/nltk_pcfg.txt',
                        help='Directory where the full pcfg is stored.')
    parser.add_argument('--output_dir', type=str, default='grammars/nltk')
    parser.add_argument('--top_k', type=float, default=0.9)
    parser.add_argument('--save', action=argparse.BooleanOptionalAction)
    parser.add_argument('--load', action=argparse.BooleanOptionalAction)
    parser.add_argument('--lexical', action=argparse.BooleanOptionalAction)
    parser.add_argument('--no_recursion', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    print('Started to load full PCFG...', flush=True)
    with open(args.pcfg_dir) as f:
        raw_grammar = f.read()
    grammar = nltk_PCFG.fromstring(raw_grammar)
    print('Finished loading PCFG...', flush=True)

    grammar = create_lookup_probs(grammar)
    prod_productions_v2 = [rule for lhs in grammar._lhs_index.values() for rule in lhs]

    subset_pcfg, subset_pcfg_pos = load_subset_pcfg(prod_productions_v2, 
                                                    args, top_k=args.top_k, 
                                                    save=args.save,
                                                    load=args.load, 
                                                    lexical=args.lexical,
                                                    no_recursion=args.no_recursion)