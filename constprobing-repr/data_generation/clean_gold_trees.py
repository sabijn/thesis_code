import argparse
from pathlib import Path
import nltk
import re


def clean_labels(input_string):
    # Define a regular expression pattern to match the labels
    pattern = r'([A-Z]+)_\d+'
    
    # Use re.sub to replace the matched pattern with only the letters
    cleaned_string = re.sub(pattern, r'\1', input_string)

    return cleaned_string


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, default=Path('corpora/eval_trees_10k.txt'))
    args = parser.parse_args()

    with open(args.data_dir / 'test_gold_trees.txt', 'r') as f:
        tree_corpus = [l.strip() for l in f]
    
    cleaned_tree_corpus = [clean_labels(tree) for tree in tree_corpus]

    with open(args.data_dir / 'gold_trees_cleaned.txt', 'w') as f:
        # f.write('\n'.join(tree._pformat_flat("", "()", False) for tree in list(cleaned_tree_corpus)))
        f.write('\n'.join([tree for tree in cleaned_tree_corpus]))
    
