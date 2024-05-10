# def test_corpus_ppl(model, corpus):
#     model.eval()

#     all_token_probs = []

#     for input_ids in tqdm(corpus):
#         sen_len = len(input_ids)
#         all_input_ids = torch.tensor(input_ids).repeat(sen_len, 1)
#         all_input_ids.fill_diagonal_(tokenizer.mask_token_id)
        
#         with torch.no_grad():
#             all_logits = model(input_ids=all_input_ids).logits
#             all_probs = all_logits.log_softmax(-1)
            
#         token_probs = all_probs[range(sen_len), range(sen_len)][range(sen_len), input_ids]
        
#         all_token_probs.extend(token_probs.tolist())

#     ppl = np.exp(-np.sum(all_token_probs)/len(all_token_probs))
    
#     return ppl

# def test_corpus_ppl_and_accuracy(model, corpus, tokenizer):
#     model.eval()

#     all_token_probs = []
#     correct_predictions = 0
#     total_predictions = 0

#     for input_ids in tqdm(corpus):
#         sen_len = len(input_ids)
#         all_input_ids = torch.tensor(input_ids).unsqueeze(0).repeat(sen_len, 1)
#         all_input_ids = all_input_ids.clone()  # to ensure we are not modifying in place
#         all_input_ids[range(sen_len), range(sen_len)] = tokenizer.mask_token_id
        
#         with torch.no_grad():
#             all_logits = model(input_ids=all_input_ids).logits
#             all_probs = all_logits.log_softmax(-1)
#             predictions = all_probs.argmax(dim=-1)
        
#         # Calculate token probabilities for perplexity
#         token_probs = all_probs[range(sen_len), range(sen_len), input_ids]
#         all_token_probs.extend(token_probs.tolist())

#         # Calculate accuracy
#         correct_predictions += (predictions[range(sen_len), range(sen_len)] == torch.tensor(input_ids)).sum().item()
#         total_predictions += sen_len

#     ppl = np.exp(-np.sum(all_token_probs) / len(all_token_probs))
#     accuracy = correct_predictions / total_predictions

#     return ppl, accuracy

import argparse
from tqdm import tqdm
import torch

from utils import *
from perplexity import masked_ppl, causal_ppl
from data import load_eval_data
import numpy as np
import pickle
import logging
import sys
import json
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

logging.basicConfig(stream=sys.stdout,
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)


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
    parser.add_argument('--size', type=int, default=None, help='Size of the dataset to evaluate on.')
    parser.add_argument('--device', default=None, choices=['cuda', 'cpu', 'mps'], help='Device to run the model on.')
    args = parser.parse_args()

    args.data_dir = f'{args.data_dir}/{args.version}'

    if args.device is None:
        if torch.cuda.is_available():
            # For running on snellius
            device = torch.device("cuda")
            logger.info('Running on cuda.')
        elif torch.backends.mps.is_available():
            # For running on M1
            device = torch.device("mps")
            logger.info('Running on M1 GPU.')
        else:
            # For running on laptop
            device = torch.device("cpu")
            logger.info('Running on CPU.')
    else:
        device = torch.device(args.device)

    all_ppls = {}
    all_acc = {}
    for top_k in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        logger.info('Evaluating top_k: %s' % top_k)
        args.top_k = top_k
        args = set_experiment_config(args)

        model, tokenizer = load_model_tokenizer(args)
        model.to(device)

        datasets = load_eval_data(args, tokenizer, args.data_dir, test_size=args.size)

        if args.model == 'babyberta':
            # perplexity methods for mlms
            ppl, all_token_probs, all_hidden_states = masked_ppl(
                model,
                datasets['test']['input_ids'], 
                device,
                mask_token_id=tokenizer.mask_token_id, 
                skip_tokens={},
                batch_size=128,
                return_hidden=True,
            )

            store_model_probs(all_token_probs, datasets, f'token_probs_eval_{args.model}_{args.version}_{args.top_k}.txt')
            bert_prob_dict = get_model_prob_dict(f'token_probs_eval_{args.model}_{args.version}_{args.top_k}.txt')
            lm_probs, all_tokens = get_probs(datasets, tokenizer, bert_prob_dict)

            
        elif args.model == 'gpt2':
            # perplexity methods for clms
            ppl, _, all_sen_probs = causal_ppl(
                model,
                datasets['test'][:100], 
                skip_tokens={tokenizer.unk_token_id},
            )
            
            pcfg_probs, lm_probs = get_causal_lm_pcfg_probs(
                "lm_training/earleyx_pcfg_dict.pickle", 
                all_sen_probs, 
                datasets['eval'][:100]['text'],
                tokenizer,
            )

        else:
            raise NotImplementedError(f"Model {args.model} not implemented")

        print(f"LM-PPL {np.exp(-np.mean(lm_probs)):.1f}")
        
        with open(f'{args.output_dir}/results_{args.model}_{args.version}_{args.top_k}.json', 'w') as f:

            json.dump({
                'lm_ppl': np.exp(-np.mean(lm_probs), f)
            })
        
        del tokenizer
        del model