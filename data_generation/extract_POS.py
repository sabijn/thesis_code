import os
import nltk.tree
import argparse
from pathlib import Path


def extract_POS(tree_corpus):
    """
    Extract POS tags from tree_corpus
    Input:
        tree_corpus: list of nltk.Tree
    Output:
        pos_corpus: list of lists of POS tags
    """
    pos_corpus = []
    for tree in tree_corpus:
        pos_corpus.append(tree.pos())

    return pos_corpus

def format_and_write(pos_corpus, output_file):
    """
    Format pos_corpus and write to output_file
    Input:
        pos_corpus: list of lists of POS tags
        output_file: str
    """
    with open(output_file, 'w') as f:
        for sentence in pos_corpus:
            for word, pos in sentence:
                f.write(f'{word} {pos.split("_")[0]}\n')
            f.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=Path('corpora/eval_trees_10k.txt'))
    parser.add_argument('--output_file', default=Path('data/train_POS.txt'))
    args = parser.parse_args()

    home = Path('/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code')
    with open(home / args.data, 'r') as f:
        tree_corpus = [nltk.tree.Tree.fromstring(l.strip()) for l in f]
    
    POS_tags = extract_POS(tree_corpus)
    format_and_write(POS_tags, home / args.output_file)

    
