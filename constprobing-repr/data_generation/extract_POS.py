import os
import nltk.tree
import argparse
from pathlib import Path
from utils import load_model
import torch

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
                # if word not in tokenizer.vocab:
                #     print('Skipping word not in tokenizer vocab: ', word)
                #     continue
                f.write(f'{word} {pos.split("_")[0]}\n')
            f.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='babyberta', choices=['deberta', 'gpt2', 'babyberta'])
    parser.add_argument('--data', default=Path('corpora/eval_trees_10k.txt'))
    parser.add_argument('--output_file', default=Path('data/train_POS_v1.txt'))
    parser.add_argument('--version', default='normal')
    parser.add_argument('--top_k', default=1.0)
    args = parser.parse_args()

    home = Path('/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code')
    if args.top_k == 1.0:
        with open(home / args.data, 'r') as f:
            tree_corpus = [nltk.tree.Tree.fromstring(l.strip()) for l in f]

        model_path = Path('/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/pcfg-lm/resources/checkpoints/deberta/')
    else:
        model_path = Path(f'/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/retrain/checkpoints/{args.model_type}/{args.version}/{args.top_k}/')

        # Check if model directory exists and find the correct checkpoint
        if not os.path.exists(model_path):
            raise ValueError(f'Model directory {model_path} does not exist.')
        
        highest_config = 0
        for dir_name in os.listdir(model_path):
            if dir_name.split('-')[0] == 'checkpoint':
                config = int(dir_name.split('-')[1])
                if config > highest_config:
                    highest_config = config

        model_path = f'{model_path}/checkpoint-{highest_config}/'

        # Load data
        with open(f'/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/reduce_grammar/corpora/{args.version}/all_trees_{args.version}_{args.top_k}.txt') as f:
            trees = [l.strip() for l in f][990_000:]
            tree_corpus = [nltk.Tree.fromstring(tree) for tree in trees]
            
    device = torch.device("cpu")
    model, tokenizer = load_model(model_path, device)
    
    POS_tags = extract_POS(tree_corpus)
    format_and_write(POS_tags, home / args.output_file)

    
