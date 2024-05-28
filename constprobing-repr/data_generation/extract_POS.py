import os
import nltk.tree
import argparse
from pathlib import Path
from utils import load_model
import torch
import json

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

def format_and_write(pos_corpus, tokenizer, output_file, word_idx=None):
    """
    Format pos_corpus and write to output_file
    Input:
        pos_corpus: list of lists of POS tags
        output_file: str
    """
    with open(output_file, 'w') as f:
        for i, sentence in enumerate(pos_corpus):
            for j, (word, pos) in enumerate(sentence):
                if word not in tokenizer.vocab:
                    print('Skipping word not in tokenizer vocab: ', word)
                    continue
                
                if i == 0 and word_idx != None:
                    if j < word_idx:
                        continue

                f.write(f'{word} {pos.split("_")[0]}\n')
            f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='babyberta', choices=['deberta', 'gpt2', 'babyberta'])
    parser.add_argument('--data_dir', default=Path('corpora/eval_trees_10k.txt'))
    parser.add_argument('--output_dir', type=Path, default=Path('data/train_POS_v1.txt'))
    parser.add_argument('--version', default='normal')
    parser.add_argument('--top_k', default=1.0)
    parser.add_argument('--specific_start', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    home = Path('/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code')

    if args.top_k == 1.0:
        with open(home / args.data_dir, 'r') as f:
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
        with open(args.output_dir / 'test_start_idx.json', 'r') as f:
            idx = json.load(f)
            sent_idx, word_idx = idx['sent_idx'], idx['word_idx']

        with open(args.data_dir) as f:
            corpus = [l.strip() for l in f][sent_idx:]

        tree_corpus = [nltk.tree.Tree.fromstring(l) for l in corpus]

        with open(args.output_dir / 'test_gold_trees.txt', 'w') as f:
            f.write('\n'.join(tree._pformat_flat("", "()", False) for tree in list(tree_corpus)))
        exit(0)
            
    device = torch.device("cpu")
    model, tokenizer = load_model(model_path, device)
    
    POS_tags = extract_POS(tree_corpus)
    format_and_write(POS_tags, tokenizer, args.output_dir / 'test_POS_labels.txt', word_idx)

    
