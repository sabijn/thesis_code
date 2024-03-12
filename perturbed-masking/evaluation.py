import numpy as np
import nltk
from collections import defaultdict, Counter
import torch
from scipy.stats import spearmanr
import pickle
from utils.ete_utils import *
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


############################################################################################################
# Evaluation from Perturbed Masking paper
############################################################################################################

def get_brackets(tree, idx=0):
    brackets = set()
    if isinstance(tree, list) or isinstance(tree, nltk.Tree):
        for node in tree:
            node_brac, next_idx = get_brackets(node, idx)
            if next_idx - idx > 1:
                brackets.add((idx, next_idx))
                brackets.update(node_brac)
            idx = next_idx
        return brackets, idx
    else:
        return brackets, idx + 1
    

def pm_constituent_evaluation(trees, results):
    prec_list = []
    reca_list = []
    f1_list = []

    nsens = 0
    for tree, result in zip(trees, results):
        nsens += 1
        _, _, _, tree2list, _ = result

        model_out, _ = get_brackets(tree)
        std_out, _ = get_brackets(tree2list)
        overlap = model_out.intersection(std_out)

        prec = float(len(overlap)) / (len(model_out) + 1e-8)
        reca = float(len(overlap)) / (len(std_out) + 1e-8)

        if len(std_out) == 0:
            reca = 1.
            if len(model_out) == 0:
                prec = 1.
        f1 = 2 * prec * reca / (prec + reca + 1e-8)
        prec_list.append(prec)
        reca_list.append(reca)
        f1_list.append(f1)

    prec_list, reca_list, f1_list \
        = np.array(prec_list).reshape((-1, 1)), np.array(reca_list).reshape((-1, 1)), np.array(
        f1_list).reshape((-1, 1))
    print('-' * 80)
    np.set_printoptions(precision=4)
    print('Mean Prec:', prec_list.mean(axis=0),
          ', Mean Reca:', reca_list.mean(axis=0),
          ', Mean F1:', f1_list.mean(axis=0))

############################################################################################################
# Evaluation from Hewitt and Manning (Spearmann)
############################################################################################################


def calc_spearman(pred_distances, gold_distances):
    """
    Based on https://github.com/john-hewitt/structural-probes/blob/master/structural-probes/reporter.py (from authors)
    """
    lengths_to_spearmanrs = defaultdict(list)
    lengths_to_pvalues = defaultdict(list)
    for i in range(len(gold_distances)):
        l = max(torch.nonzero(gold_distances[i]!=-1, as_tuple=True)[0]) + 1
        if l == 1:
            continue
        predictions = pred_distances[i][:l,:l]
        labels = gold_distances[i][:l,:l]
        spearmanrs = [spearmanr(pred, gold) for pred, gold in zip(predictions, labels)]
        lengths_to_spearmanrs[int(l)].extend([x.correlation for x in spearmanrs])
        lengths_to_pvalues[int(l)].extend([x.pvalue for x in spearmanrs])

    mean_spearman_for_each_length = {length: np.mean(lengths_to_spearmanrs[length]) for length in lengths_to_spearmanrs}
    mean_pvalue_for_each_length = {length: np.mean(lengths_to_pvalues[length]) for length in lengths_to_pvalues}
    mean = np.mean([mean_spearman_for_each_length[x] for x in range(5,51) if x in mean_spearman_for_each_length])
    mean_pvalue = np.mean([mean_pvalue_for_each_length[x] for x in range(5,51) if x in mean_pvalue_for_each_length])

    return mean, mean_pvalue


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    # Data args
    parser.add_argument('--model', default='deberta')
    parser.add_argument('--metric', default='dist', choices=['dist', 'cos'])
    parser.add_argument('--layer', default=8 , type=str, choices=['0', '1', '2', '3', '4', '5', '6', '7', '8'])

    args = parser.parse_args()

        # Load the gold trees
    with open('/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/corpora/eval_trees_1000.txt', 'r') as f:
        gold_trees = [line.strip('\n') for line in f.readlines()]
    
    for tree in gold_trees:
        #tree = tree.replace('(DOT_0 .)', '')
        nltk.Tree.fromstring(tree).pretty_print()
        print(gold_tree_to_ete(tree))
        break
    exit()
    
    gold_ete_trees = [gold_tree_to_ete(tree) for tree in gold_trees]
    gold_distances = create_distances(gold_ete_trees)

    results_dict = {'deberta' : [], 'gpt2' : []}
    for m in tqdm(['deberta', 'gpt2']):
        for l in tqdm(['0', '1', '2', '3', '4', '5', '6', '7', '8'], leave=False):
            # Load the predicted trees
            with open(f'results/trees/trees_{m}_{args.metric}_{l}.txt', 'r') as f:
                pred_trees = [line.strip('\n') for line in f.readlines()]

            pred_ete_trees = [create_ete3_from_pred(tree) for tree in pred_trees]
            pred_distances = create_distances(pred_ete_trees)

            # Check if the tree lengths are the same, if not, remove tree
            counter = 0
            for i, (pred, gold) in enumerate(zip(pred_distances, gold_distances)):
                if pred.shape != gold.shape:
                    logger.warning(f'The tree length of sentence {i} ({pred.shape[0]}) is not the same as the gold tree length ({gold.shape[0]}).\
                                Removing sentence from evaluation.')
                    del pred_distances[i]
                    if m == 'deberta' and l == '0':
                        del gold_distances[i]

            # Calculate Spearman correlation
            spearman, pvalue = calc_spearman(pred_distances, gold_distances)
            results_dict[m].append((spearman, pvalue))
            print(f'Spearman correlation: {spearman}')
    
    # Save results
    with open(f'results/spearman_dist_all.pkl', 'wb') as f:
        pickle.dump(results_dict, f)
    
    
