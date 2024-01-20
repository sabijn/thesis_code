from pathlib import Path
import nltk

data_path = Path('/Users/sperdijk/Documents/Master/Jaar_3/Thesis/PTB/penn-wsj-line.txt')


with open(data_path) as f:
    lines = f.readlines()
    f.close()
