import argparse
from tqdm import tqdm
import torch

from utils import set_experiment_config, load_model_tokenizer
from data import load_data
import numpy as np
import pickle


def test_corpus_ppl(model, corpus):
    model.eval()

    all_token_probs = []

    for input_ids in tqdm(corpus):
        sen_len = len(input_ids)
        all_input_ids = torch.tensor(input_ids).repeat(sen_len, 1)
        all_input_ids.fill_diagonal_(tokenizer.mask_token_id)
        
        with torch.no_grad():
            all_logits = model(input_ids=all_input_ids).logits
            all_probs = all_logits.log_softmax(-1)
            
        token_probs = all_probs[range(sen_len), range(sen_len)][range(sen_len), input_ids]
        
        all_token_probs.extend(token_probs.tolist())

    ppl = np.exp(-np.sum(all_token_probs)/len(all_token_probs))
    
    return ppl


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the model')
    parser.add_argument('--model', type=str, default='babyberta', choices=['gpt2', 'babyberta'],
                        help='Model to evaluate')
    parser.add_argument('--model_dir', type=str, default='checkpoints')
    parser.add_argument('--output_dir', type=str, default='/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/retrain/results')
    parser.add_argument('--top_k', type=float, default=0.2, choices=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument('--version', type=str, default='normal', choices=['normal', 'lexical', 'pos'],
                        help='Version of the corpus to evaluate.')
    parser.add_argument('--data_dir', type=str, default='/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/reduce_grammar/corpora')
    args = parser.parse_args()

    all_ppls = []
    for top_k in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        args.top_k = top_k
        args = set_experiment_config(args)

        model, tokenizer = load_model_tokenizer(args)
        datasets = load_data(args, tokenizer, args.data_dir, train_size=0, dev_size=0)
        all_ppls.append(test_corpus_ppl(model, datasets['test']['input_ids']))
    
    with open(f'{args.output_dir}/ppls_{args.model}_{args.version}.pkl', 'wb') as f:
        pickle.dump(all_ppls, f)