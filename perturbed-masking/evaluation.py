import numpy as np
import nltk
from collections import defaultdict
import torch
from scipy.stats import spearmanr
import pickle
from utils.ete_utils import tree2etetree, create_gold_distances, create_pred_distances

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
    for i in range(len(gold_distances)):
        l = max(torch.nonzero(gold_distances[i]!=-1, as_tuple=True)[0]) + 1
        if l == 1:
            continue
        predictions = pred_distances[i][:l,:l]
        labels = gold_distances[i][:l,:l]
        spearmanrs = [spearmanr(pred, gold) for pred, gold in zip(predictions, labels)]
        lengths_to_spearmanrs[int(l)].extend([x.correlation for x in spearmanrs])

    mean_spearman_for_each_length = {length: np.mean(lengths_to_spearmanrs[length]) for length in lengths_to_spearmanrs}
    mean = np.mean([mean_spearman_for_each_length[x] for x in range(5,51) if x in mean_spearman_for_each_length])

    return mean


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    # Data args
    parser.add_argument('--model', default='deberta')
    parser.add_argument('--metric', default='dist', choices=['dist', 'cos'])
    parser.add_argument('--layer', default=8 , type=str, choices=['0', '1', '2', '3', '4', '5', '6', '7', '8'])

    args = parser.parse_args()

    # Load the gold distances
    with open(f'results/trees/trees_{args.model}_{args.metric}_{args.layer}_list.pkl', 'rb') as f:
        pred_trees = pickle.load(f)

    pred_trees = [tree2etetree(tree) for tree in pred_trees]
    pred_distances = create_pred_distances(pred_trees)
    print(pred_distances[0])


    # # Load the predicted distances
    # with open(f'results/distances/{args.model}_{args.mertric}_layer_{args.layer}.pkl', 'rb') as f:
    #     gold_trees = pickle.load(f)

    # Calculate the Spearman correlation