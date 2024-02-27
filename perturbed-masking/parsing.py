import nltk
import argparse
import pickle
import numpy as np
from decoder import mart, right_branching, left_branching
import re
word_tags = ['RBR', 'DT', 'VBP', 'VBZ', 'IN', 'VBG', 'NNS', 'CC', 'FW', 'VBD', 'HASH', 'RBS', 'MD', 'DOT', 'RP', 'POS', 'EX', 'TO', 'NNPS', 'PDT', 'VBN', 'VB', 'RB', 'JJR', 'PRPDOLLAR', 'JJ', 'APOSTROPHE', 'RRB', 'JJS', 'SYM', 'WPDOLLAR', 'COLON', 'UH', 'WDT', 'PRP', 'TICK', 'LRB', 'WRB', 'WP', 'NN', 'COMMA', 'CD', 'NNP']
from collections import Counter
from utils import match_tokenized_to_untokenized
import os

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


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def merge_subwords_in_one_row(matrix, mapping):
    # merge subwords in one row
    merge_column_matrix = []
    for i, line in enumerate(matrix):
        new_row = []
        buf = []
        for j in range(0, len(line) - 1):
            buf.append(line[j])
            if mapping[j] != mapping[j + 1]:
                new_row.append(buf[0])
                buf = []
        merge_column_matrix.append(new_row)
    
    return merge_column_matrix


def merge_subwords_in_one_column(matrix, mapping):
    # merge subwords in one column (influence on subwords == average of the subwords)
    # transpose the matrix so we can work with row instead of columns
    matrix = np.array(matrix).transpose()
    matrix = matrix.tolist()
    final_matrix = []
    for i, line in enumerate(matrix):
        new_row = []
        buf = []
        for j in range(0, len(line) - 1):
            buf.append(line[j])
            if mapping[j] != mapping[j + 1]:
                if args.subword == 'max':
                    new_row.append(max(buf))
                elif args.subword == 'avg':
                    new_row.append((sum(buf) / len(buf)))
                elif args.subword == 'first':
                    new_row.append(buf[0])
                buf = []
        final_matrix.append(new_row)

    return np.array(final_matrix).transpose()


def matrix_transform(matrix):
    matrix = matrix[1:, 1:]
    matrix = softmax(matrix)
    matrix = 1. - matrix
    np.fill_diagonal(matrix, 0.)

    return matrix


def decoding(args):
    """
    Map the impact matrix to an unlabeled parse tree
    """
    # impact matrix
    with open(args.matrix, 'rb') as f:
        results = pickle.load(f)

    trees = []
    for (sen, tokenized_text, init_matrix, tree2list, nltk_tree) in results:
        mapping = match_tokenized_to_untokenized(tokenized_text, sen)

        # merge subwords in one row (influence by subwords == influence first subword)
        merge_column_matrix = merge_subwords_in_one_row(init_matrix, mapping)

        # merge subwords in one column (influence on subwords == average of the subwords)
        final_matrix = merge_subwords_in_one_column(merge_column_matrix, mapping)
        assert final_matrix.shape[0] == final_matrix.shape[1]
        assert final_matrix.shape[0] == init_matrix.shape[0] - 1

        final_matrix = matrix_transform(final_matrix)

        if args.decoder == 'mart':
            parse_tree = mart(final_matrix, sen)
            trees.append(parse_tree)

        if args.decoder == 'right_branching':
            trees.append(right_branching(sen))

        if args.decoder == 'left_branching':
            trees.append(left_branching(sen))

    return trees, results


def constituent_evaluation(trees, results):
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data args
    parser.add_argument('--model', default='deberta')
    parser.add_argument('--mertric', default='dist', choices=['dist', 'cos'])
    parser.add_argument('--layer', default=8 , type=str, choices=['0', '1', '2', '3', '4', '5', '6', '7', '8'])

    # Decoding args
    parser.add_argument('--decoder', default='mart')
    parser.add_argument('--subword', default='avg')

    args = parser.parse_args()
    args.matrix = f'results/{args.model}_{args.mertric}_{args.layer}.pkl'

    # check if file exists
    if not os.path.exists(args.matrix):
        print(f'File {args.matrix} does not exist')
        raise FileNotFoundError

    trees, results = decoding(args)
    print('pred', trees[0])
    print('gold', results[0][3])
    exit()

    constituent_evaluation(trees[:10], results[:10])