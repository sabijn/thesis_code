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
def calculate_sentence_bigram_probability(sentence, bigram_probabilities):
    bigram_probs = []
    # Multiply by each bigram probability in the sentence
    for i in range(1, len(sentence)):
        word1, word2 = sentence[i - 1], sentence[i]

        bigram_prob = bigram_probabilities.get((word1, word2), 0)
        bigram_probs.append(bigram_prob)
    
    return bigram_probs

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

    # Step 1: Count occurrences of unigrams and bigrams
    unigram_counts = Counter()
    bigram_counts = defaultdict(Counter)

    for sentence in corpus:
        sentence.insert(0, '<s>')
        # Update unigram counts
        unigram_counts.update(sentence)
        
        # Update bigram counts
        for i in range(len(sentence) - 1):
            bigram_counts[sentence[i]][sentence[i + 1]] += 1

    # Step 2: Calculate bigram probabilities
    bigram_probabilities = {}
    for word1, following_words in bigram_counts.items():
        total_count_word1 = unigram_counts[word1]
        for word2, count in following_words.items():
            bigram_probabilities[(word1, word2)] = np.log(count / total_count_word1)

    bigram_sentence_probabilities = {}

    for sentence in corpus:
        bigram_sentence_prob = calculate_sentence_bigram_probability(sentence, bigram_probabilities)
        assert len(sentence) - 1 == len(bigram_sentence_prob), f'{len(sentence)} {len(bigram_sentence_prob)}'
        bigram_sentence_probabilities[' '.join(sentence[1:])] = bigram_sentence_prob

    assert len(corpus) == len(bigram_sentence_probabilities)
    # write to json file
    with open('pcfg_probs/bigram/pcfg_mlm_probs_bigram.json', 'w') as f:
        json.dump(bigram_sentence_probabilities, f, indent=2)

    