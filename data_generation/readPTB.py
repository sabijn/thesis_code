from pathlib import Path
import nltk

data_path = Path('/Users/sperdijk/Documents/Master/Jaar_3/Thesis/PTB/penn-wsj-line.txt')
results_path = Path('/Users/sperdijk/Documents/Master/Jaar_3/Thesis/PTB/penn-line.txt')
sentences_results_path = Path('/Users/sperdijk/Documents/Master/Jaar_3/Thesis/PTB/penn-sentences.txt')
# results_path = Path('/Users/sperdijk/Documents/Master/Jaar_3/Thesis/PTB/penn-sentences.txt')

results_file = open(results_path, 'w')
sentence_file = open(sentences_results_path, 'w')
with open(data_path) as f:
    for tree in f.readlines():
        if tree == '\n':
            continue

        stripped_tree = tree.replace('(`` ``)', '').replace("('' '')", '')
        results_file.write(stripped_tree)
        sentence_file.write(" ".join(nltk.Tree.fromstring(stripped_tree).leaves()) + '\n')



