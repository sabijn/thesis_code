import argparse
from pathlib import Path
import json
import numpy as np
from collections import defaultdict
import torch

def load_mlm_pcfg_probs(filename, log_probs=True):
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
            current_probabilities.append(np.log(float(line)) if log_probs else float(line))
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

def load_model_probs(filename):
    with open(filename, 'r') as file:
        data = file.readlines()

    result = {}
    current_sentence = ""
    current_probabilities = []
    
    for line in data:
        line = line.strip()
        
        try:
            current_probabilities.append(float(line))  # Add the probability
        except ValueError:
            if current_sentence and current_probabilities:
                result[current_sentence] = current_probabilities
                assert len(current_sentence.split()) == len(current_probabilities), f'{current_sentence}, {current_probabilities}'
            current_sentence = line  # Start a new sentence
            current_probabilities = []

            continue

    # Add the last sentence and its probabilities
    if current_sentence and current_probabilities:
        result[current_sentence] = current_probabilities

    return result

def load_pos_tags(filename):
    with open(filename, 'r') as file:
        data = file.readlines()

    result = {}
    current_sentence = ""
    current_tags = []

    for line in data:
        line = line.strip()

        if not line:  # Empty line signifies a new sentence
            if current_sentence:  # Only store if we have tokens from the last sentence
                result[current_sentence] = current_tags
                current_tags = []  # Reset for the next sentence
                current_sentence = ""
        else:
            # Split the line into token and tag (first and second column)
            token, tag = line.split()
            if not current_sentence:
                current_sentence = token
            else:
                current_sentence += " " + token
            current_tags.append(tag)

    # Add the last sentence if there's no trailing newline at the end of the file
    if current_sentence:
        result[current_sentence] = current_tags

    return result

def get_mean_divergence(pcfg_probs, model_probs_per_layer, mapping, pos_tag_dict):
    # load from txt file
    with open(f'/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/pcfg-lm/resources/checkpoints/deberta/vocab.txt', 'r') as f:
        vocab = f.readlines()
        vocab = [word.strip() for word in vocab]

    # Initialize dictionaries to track the sum and count of differences
    sum_diff = defaultdict(lambda: defaultdict(float))
    count_diff = defaultdict(lambda: defaultdict(int))

    print('Start calculating differences...')
    for layer, model_probs in model_probs_per_layer.items():
        for sentence, model_sent_prob in model_probs.items():
            pcfg_probs_sentence = pcfg_probs[sentence]
            pos_tags = pos_tag_dict[sentence]

            for i, (word, model_prob, pcfg_prob, pos_tag) in enumerate(zip(sentence.split(" "), model_sent_prob, pcfg_probs_sentence, pos_tags)):
                if word in vocab:
                    class_label = mapping[pos_tag]
                    
                    # Accumulate the difference and increase the count
                    sum_diff[layer][class_label] += (model_prob - pcfg_prob)
                    count_diff[layer][class_label] += 1

    # Calculate the mean difference for each (layer, class_label) pair
    print(count_diff)
    mean_div = defaultdict(dict)
    for layer, class_sums in sum_diff.items():
        for class_label, total_diff in class_sums.items():
            mean_div[layer][class_label] = total_diff / count_diff[layer][class_label]


    return mean_div

def convert_log_prob_to_prob(log_probs):
    temp = {}
    for sent, prob in log_probs.items():
        # convert all floats in list to probabilities
        temp[sent] = [np.exp(x) for x in prob]

    return temp

def get_layer_diffs(model_probs_per_layer, mapping, pos_tag_dict):
    # load from txt file
    with open(f'/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/pcfg-lm/resources/checkpoints/deberta/vocab.txt', 'r') as f:
        vocab = f.readlines()
        vocab = [word.strip() for word in vocab]

    # Initialize dictionaries to track the sum and count of differences
    sum_diff = defaultdict(lambda: defaultdict(float))
    count_diff = defaultdict(lambda: defaultdict(int))

    print('Start calculating differences...')
    for layer, model_probs in model_probs_per_layer.items():
        for sentence, model_sent_prob in model_probs.items():
            last_sent_probs = model_probs_per_layer[8][sentence]
            pos_tags = pos_tag_dict[sentence]

            for i, (word, model_prob, last_prob, pos_tag) in enumerate(zip(sentence.split(" "), model_sent_prob, last_sent_probs, pos_tags)):
                if word in vocab:
                    class_label = mapping[pos_tag]
                    
                    # Accumulate the difference and increase the count
                    sum_diff[layer][class_label] += (model_prob - last_prob)
                    count_diff[layer][class_label] += 1

    # Calculate the mean difference for each (layer, class_label) pair
    mean_div = defaultdict(dict)
    for layer, class_sums in sum_diff.items():
        for class_label, total_diff in class_sums.items():
            mean_div[layer][class_label] = total_diff / count_diff[layer][class_label]

    return mean_div

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mechanistic Interpretation')

    parser.add_argument('--model', type=str, default='deberta', choices=['deberta', 'gpt2'])
    parser.add_argument('--top_k', type=float, default=1.0, choices=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    parser.add_argument('--data_dir', type=Path, default='/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/mechanistic-interpret/results')
    parser.add_argument('--pcfg_dir', type=Path, default='/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/mechanistic-interpret/pcfg_probs')
    parser.add_argument('--output_dir', type=Path, default='/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/mechanistic-interpret/results/diffs')
    parser.add_argument('--mapping_file', type=Path, default='/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/mechanistic-interpret/pos_tags/mapping.json')
    parser.add_argument('--pos_tags', type=Path, default='/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/mechanistic-interpret/pos_tags/train_POS_v1.txt')
    parser.add_argument('--log_probs', type=str, default='False', choices=['True', 'False'])
    parser.add_argument('--ngram', type=str, default='normal', choices=['normal', 'unigram', 'bigram', 'uniform'])

    args = parser.parse_args()
    args.log_probs = bool(eval(args.log_probs))
    
    if args.log_probs:
        print('With logprobs')
        data_file = args.data_dir / args.model / str(args.top_k) / 'log_probs'
        args.output_dir = args.output_dir / 'log_probs'
    else:
        print('With regular probs')
        data_file = args.data_dir / args.model / str(args.top_k) / 'probs'
        args.output_dir = args.output_dir / 'probs'

    if args.model == 'gpt2':
        with open(args.pcfg_dir / f'pcfg_clm_probs.json', 'r') as f:
            pcfg_probs = json.load(f) # note these are negative log probs
            pcfg_probs = convert_log_prob_to_prob(pcfg_probs)

    elif args.model == 'deberta':
        if args.ngram == 'bigram':
            with open(args.pcfg_dir / f'pcfg_mlm_probs_bigram.json', 'r') as f:
                pcfg_probs = json.load(f)

        elif args.ngram == 'unigram':
            with open(args.pcfg_dir / f'pcfg_mlm_probs_unigram.json', 'r') as f:
                pcfg_probs = json.load(f)

        elif args.ngram == 'normal':
            pcfg_probs = load_mlm_pcfg_probs(args.pcfg_dir / f'pcfg_mlm_probs.txt', args.log_probs)

        elif args.ngram == 'uniform':
            with open(args.pcfg_dir / f'pcfg_mlm_probs_uniform.json', 'r') as f:
                pcfg_probs = json.load(f)
        else:
            raise NotImplementedError(f'Ngram {args.ngram} not supported.')
    
    model_probs_per_layer = {}

    for layer in range(9):
        model_probs_per_layer[layer] = load_model_probs(data_file / f'token_probs_eval_{args.model}_{args.top_k}_layer_{layer}.txt')

    ########### Hacky shit
    # with open(f'/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/pcfg-lm/resources/checkpoints/deberta/vocab.txt', 'r') as f:
    #     vocab = f.readlines()
    #     vocab = [word.strip() for word in vocab]
    
    # total = 0
    # for (sentence, probs) in pcfg_probs.items():
    #     for i, (word, prob) in enumerate(zip(sentence.split(" "), probs)):
    #         if word in vocab:
    #             total += 1
    # print("Total PCFG, ", total)

    # total = 0
    # for (sentence, probs) in model_probs_per_layer[8].items():
    #     for i, (word, prob) in enumerate(zip(sentence.split(" "), probs)):
    #         if word in vocab:
    #             total += 1
    # print("Total model, ", total)

    # all_pcfg_probs = []
    # all_model_probs = []
    # print('model - pcfg', set(model_probs_per_layer[8].keys()) - set(pcfg_probs.keys()))
    # print('pcfg - model', set(pcfg_probs.keys()) - set(model_probs_per_layer[8].keys()))
    # for sentence, model_sent_prob in model_probs_per_layer[8].items():
    #     pcfg_probs_sentence = pcfg_probs[sentence]

    #     for i, (word, model_prob, pcfg_prob) in enumerate(zip(sentence.split(" "), model_sent_prob, pcfg_probs_sentence)):
    #         if word in vocab:
    #             all_pcfg_probs.append(pcfg_prob)
    #             all_model_probs.append(model_prob)

    # tensor_pcfg_probs = torch.tensor(all_pcfg_probs)
    # tensor_model_probs = torch.tensor(all_model_probs)
    # print('Model probs length', tensor_model_probs.shape)
    # print('PCFG probs length', tensor_pcfg_probs.shape)
    # print('Min PCFG', tensor_pcfg_probs.min())
    # print('Min Model', tensor_model_probs.min())
    # print('diff', torch.mean(tensor_model_probs - tensor_pcfg_probs))
    # exit(1)
    ##############

    assert len(pcfg_probs) == len(model_probs_per_layer[0]), f'{len(pcfg_probs)}, {len(model_probs_per_layer[0])}'
    
    # load json from path
    with open(args.mapping_file, 'r') as f:
        mapping = json.load(f)

    pos_tag_dict = load_pos_tags(args.pos_tags)
    
    # per_layer_diff = get_layer_diffs(model_probs_per_layer,
    #                                  mapping,
    #                                  pos_tag_dict)
    # with open(args.output_dir / outputname, 'w') as f:
    #     json.dump(per_layer_diff, f)

    mean_div = get_mean_divergence(pcfg_probs, 
                                   model_probs_per_layer, 
                                   mapping, 
                                   pos_tag_dict)
    
    if args.ngram == 'bigram':
        outputname = f'mean_divergence_{args.model}_{args.top_k}_bigram.json'
    elif args.ngram == 'unigram':
        outputname = f'mean_divergence_{args.model}_{args.top_k}_unigram.json'
    elif args.ngram == 'normal':
        outputname = f'mean_divergence_{args.model}_{args.top_k}.json'
    elif args.ngram == 'uniform':
        outputname = f'mean_divergence_{args.model}_{args.top_k}_uniform.json'
    else:
        raise NotImplementedError(f'Ngram {args.ngram} not supported.')
    
    with open(args.output_dir / outputname, 'w') as f:
        json.dump(mean_div, f)

    



