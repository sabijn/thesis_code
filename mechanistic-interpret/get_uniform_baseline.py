import argparse
import json
from pathlib import Path
import numpy as np

from collections import defaultdict, Counter

def load_mlm_pcfg_probs(filename):
    with open(filename, 'r') as file:
        data = file.readlines()

    result = {}
    current_sentence = ""
    current_probabilities = []
    
    for line in data:
        line = line.strip()
        
        if line.startswith("Token:"):  # Skip token lines
            continue
        
        try:
            current_probabilities.append(np.log(float(line)))  # Add the probability
        except ValueError:
            if current_sentence and current_probabilities:
                result[current_sentence] = current_probabilities
                assert len(current_sentence.split()) == len(current_probabilities)

            line = line.replace("'", "<apostrophe>")

            current_sentence = line  # Start a new sentence
            current_probabilities = []

            continue

    # Add the last sentence and its probabilities
    if current_sentence and current_probabilities:
        result[current_sentence] = current_probabilities

    return result

# Function to calculate bigram probability of a sentence
def calculate_sentence_bigram_probability(sentence, unigram_probabilities):
    unigram_probs = []

    for i in range(len(sentence)):
        unigram_prob = unigram_probabilities.get(sentence[i], 0)
        unigram_probs.append(unigram_prob)
    
    return unigram_probs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pcfg_dir', type=Path, default='/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/mechanistic-interpret/pcfg_probs')

    parser.add_argument('--model', type=str, default='deberta', choices=['deberta', 'gpt2'])
    args = parser.parse_args()

    if args.model == 'gpt2':
        with open(args.pcfg_dir / f'pcfg_clm_probs.json', 'r') as f:
            pcfg_probs = json.load(f) # note these are negative log probs

    elif args.model == 'deberta':
        pcfg_probs = load_mlm_pcfg_probs(args.pcfg_dir / f'pcfg_mlm_probs.txt')
    
    else:
        raise ValueError(f'Model {args.model} not supported.')

    # Sample corpus (list of tokenized sentences)
    corpus = [sent.split(" ") for sent in pcfg_probs.keys()]

    # Step 1: Count occurrences of unigrams
    unigram_counts = Counter()

    for sentence in corpus:
        unigram_counts.update(sentence)  # Update unigram counts

    # Step 2: Calculate total count of all unigrams
    total_unigrams = sum(unigram_counts.values())

    # Step 3: Calculate unigram probabilities
    unigram_probabilities = {}
    for word, count in unigram_counts.items():
        unigram_probabilities[word] = np.log(1 / total_unigrams)

    unigram_sentence_probabilities = {}

    for sentence in corpus:
        unigram_sentence_prob = calculate_sentence_bigram_probability(sentence, unigram_probabilities)
        assert len(sentence) == len(unigram_sentence_prob), f'{len(sentence)} {len(unigram_sentence_prob)}'
        unigram_sentence_probabilities[' '.join(sentence)] = unigram_sentence_prob

    assert len(corpus) == len(unigram_sentence_probabilities)
    # write to json file
    with open('pcfg_probs/uniform/pcfg_mlm_probs_uniform.json', 'w') as f:
        json.dump(unigram_sentence_probabilities, f, indent=2)

    