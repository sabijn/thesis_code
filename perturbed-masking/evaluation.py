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
    
    return (prec_list.mean(axis=0), reca_list.mean(axis=0), f1_list.mean(axis=0))


def classic_evaluation(args, gold_trees, pred_trees):
    all_layer_results = []
    for layer in range(args.layers):
        gold, pred = gold_trees[layer], pred_trees[layer]

        skip_idx = []
        for i, result in enumerate(gold):
            gold_tree = result[3]
            if gold_tree[-1] not in ['.', '!', '?']:
                skip_idx.append(i)

        # remove list of indices from list
        pred = [pred[i] for i in range(len(pred)) if i not in skip_idx]

        gold = [gold[i] for i in range(len(gold)) if i not in skip_idx]
        assert len(pred_trees) == len(gold_trees)

        results = pm_constituent_evaluation(pred, gold)
        all_layer_results.append(results)
    
    result_path = f'results/classic_dist_{args.model}_all_layers_without_punct.pkl' if args.remove_punct else f'results/classic_dist_{args.model}_all_layers.pkl'
    with open(result_path, 'wb') as f:
        pickle.dump(all_layer_results, f)


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
    

def spearman_evaluation(args, pred_trees):
    # Load the gold trees
    with open(args.data, 'r') as f:
        gold_trees = [line.strip('\n') for line in f.readlines()]

    if args.remove_punct:
        # trees are skipped if they have "'" or "''" as last token
        gold_trees, skipped_idx = remove_punctuation_nltk(gold_trees)
    else:
        skipped_idx = []

    gold_ete_trees = [gold_tree_to_ete(tree) for tree in gold_trees]
    gold_distances = create_distances(gold_ete_trees)

    results = []
    for l in tqdm(range(args.layers), leave=False):
        pred_layer = [tree for i, tree in enumerate(pred_trees[l]) if i not in skipped_idx]

        assert len(pred_layer) == len(gold_trees), f'Length of pred trees ({len(pred_layer)}) is not the same as the gold trees ({len(gold_trees)}).'
        pred_ete_trees = [create_ete3_from_pred(tree) for tree in pred_layer]
        pred_distances = create_distances(pred_ete_trees)

        # Check if the tree lengths are the same, if not, remove tree
        for i, (pred, gold) in enumerate(zip(pred_distances, gold_distances)):
            if pred.shape != gold.shape:
                logger.warning(f'The tree length of sentence {i} ({pred.shape[0]}) is not the same as the gold tree length ({gold.shape[0]}).\
                            Removing sentence from evaluation.')
                del pred_distances[i]
                if l == '0':
                    del gold_distances[i]

        # Calculate Spearman correlation
        spearman, pvalue = calc_spearman(pred_distances, gold_distances)
        results.append((spearman, pvalue))
        print(f'Spearman correlation: {spearman}')
    
    # Save results
    result_path = f'results/spearman_{args.metric}_{args.model}_without_punct.pkl' if args.remove_punct else f'results/spearman_{args.metric}_{args.model}.pkl'
    with open(result_path, 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    # Data args
    parser.add_argument('--model', default='deberta')
    parser.add_argument('--metric', default='dist', choices=['dist', 'cos'])
    parser.add_argument('--layer', default=8 , type=str, choices=['0', '1', '2', '3', '4', '5', '6', '7', '8'])
    parser.add_argument('--evaluation', default='spearman', choices=['spearman', 'classic'])
    parser.add_argument('--remove_punct', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    if args.evaluation == 'spearman':
        spearman_evaluation(args)
    elif args.evaluation == 'classic':
        classic_evaluation(args)
        
    
    
