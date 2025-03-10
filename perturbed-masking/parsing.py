import argparse
import pickle
import numpy as np
import os

from decoder import mart, right_branching, left_branching
from evaluation import pm_constituent_evaluation
from utils.utils import match_tokenized_to_untokenized
from utils.wr_utils import listtree2str


WORD_TAGS = ['RBR', 'DT', 'VBP', 'VBZ', 'IN', 'VBG', 'NNS', 'CC', 'FW', 'VBD', 'HASH', 'RBS', 'MD', 'DOT', 'RP', 'POS', 'EX', 'TO', 'NNPS', 'PDT', 'VBN', 'VB', 'RB', 'JJR', 'PRPDOLLAR', 'JJ', 'APOSTROPHE', 'RRB', 'JJS', 'SYM', 'WPDOLLAR', 'COLON', 'UH', 'WDT', 'PRP', 'TICK', 'LRB', 'WRB', 'WP', 'NN', 'COMMA', 'CD', 'NNP']


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


def merge_subwords_in_one_column(args, matrix, mapping):
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


def matrix_transform(matrix, remove_punct=False):
    matrix = softmax(matrix)
    matrix = 1. - matrix
    np.fill_diagonal(matrix, 0.)

    return matrix

def remove_punctuation(sentence, matrix):
    # remove CLS tokens
    matrix = matrix[1:, 1:]

    # remove punctuation from sentence
    if sentence[-1] in ['.', '!', '?']:
        sentence = sentence[:-1]
        matrix = matrix[:-1, :-1]
    
    return sentence, matrix

def decoding(args, results):
    """
    Map the impact matrix to an unlabeled parse tree
    """
    trees = []
    for (sen, tokenized_text, init_matrix, tree2list, nltk_tree) in results:
        mapping = match_tokenized_to_untokenized(tokenized_text, sen)

        # merge subwords in one row (influence by subwords == influence first subword)
        merge_column_matrix = merge_subwords_in_one_row(init_matrix, mapping)

        # merge subwords in one column (influence on subwords == average of the subwords)
        final_matrix = merge_subwords_in_one_column(args, merge_column_matrix, mapping)
        assert final_matrix.shape[0] == final_matrix.shape[1]
        assert final_matrix.shape[0] == init_matrix.shape[0] - 1

        # removing punctuation from matrix and sentence
        if args.remove_punct:
            sen, final_matrix = remove_punctuation(sen, final_matrix)
            assert len(sen) == final_matrix.shape[0]
        else:
            final_matrix = final_matrix[1:, 1:]
        
        # creating probailities
        final_matrix = matrix_transform(final_matrix)

        if args.decoder == 'mart':
            parse_tree = mart(final_matrix, sen)
            trees.append(parse_tree)

        if args.decoder == 'right_branching':
            trees.append(right_branching(sen))

        if args.decoder == 'left_branching':
            trees.append(left_branching(sen))

    return trees, results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data args
    parser.add_argument('--model', default='deberta', choices=['deberta', 'gpt2'])
    parser.add_argument('--mertric', default='dist', choices=['dist', 'cos'])

    # Decoding args
    parser.add_argument('--decoder', default='mart')
    parser.add_argument('--subword', default='avg')
    parser.add_argument('--remove_punct', action=argparse.BooleanOptionalAction)
    parser.add_argument('--list', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()


    # check if file exists
    for i in range(1, 9):
        args.matrix = f'results/i_matrices/{args.model}_{args.mertric}_{i}.pkl'
        if not os.path.exists(args.matrix):
            raise FileNotFoundError(f'File {args.matrix} does not exist.')

        trees, results = decoding(args)

        if args.remove_punct:
            results_dir = 'trees_without_punct'
        else:
            results_dir = 'trees'
        
        if args.list:
            with open(f'results/{results_dir}/trees_{args.model}_{args.mertric}_{i}_list.pkl', 'wb') as f:
                pickle.dump(trees, f)
        else:
            # convert trees to strings and write to file
            trees_str = [listtree2str(tree) for tree in trees]
            with open(f'results/{results_dir}/trees_{args.model}_{args.mertric}_{i}.txt', 'w') as f:
                for tree in trees_str:
                    f.write(f'{tree}\n')
