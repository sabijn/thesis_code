import argparse
from tqdm import tqdm
import torch

from utils import *
from perplexity import masked_ppl, causal_ppl, pcfg_perplexity
from data import load_eval_data
import numpy as np
import pickle
import logging
import sys
import json
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

from classes import (TokenizerConfig, 
                        Tokenizer, 
                        PCFGConfig,
                        PCFG)

logging.basicConfig(stream=sys.stdout,
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

def load_language(args, encoder="transformer", corpus_size=200_000):
    if args.hardware == 'snellius':
        grammar_file = f'/scratch-shared/sabijn/{args.version}/subset_pcfg_{args.top_k}.txt'
        corpus_file = f'/scratch-shared/sabijn/{args.version}/corpus_{args.top_k}_{args.version}.pt'
    
    elif args.hardware == 'local':
        grammar_file = f'/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/reduce_grammar/grammars/nltk/{args.version}/subset_pcfg_{args.top_k}.txt'
        corpus_file = f'/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/reduce_grammar/corpora/{args.version}/corpus_{args.top_k}_{args.version}.pt'
    
    else:
        raise NotImplementedError(args.hardware)


    tokenizer_config = TokenizerConfig(
            add_cls=(encoder == "transformer"),
            masked_lm=(encoder == "transformer"),
            unk_threshold=5,
        )
    
    tokenizer = Tokenizer(tokenizer_config)

    config = PCFGConfig(
        is_binary=False,
        min_length=6,
        max_length=25,
        max_depth=25,
        corpus_size=corpus_size,
        grammar_file=grammar_file,
        start="S_0",
        masked_lm=(encoder == "transformer"),
        allow_duplicates=True,
        split_ratio=(0.8,0.1,0.1),
        use_unk_pos_tags=True,
        verbose=True,
        store_trees=True,
        output_dir='.',
        top_k=args.top_k,
        version=args.version,
        file=corpus_file
    )

    language = PCFG(config, tokenizer)

    return language, tokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the model')
    parser.add_argument('--model', type=str, default='babyberta', choices=['gpt2', 'babyberta'],
                        help='Model to evaluate')
    parser.add_argument('--model_dir', type=str, default='checkpoints')
    parser.add_argument('--output_dir', type=str, default='/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/retrain/results')
    parser.add_argument('--output_file_pcfg', type=str, default='/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/reduce_grammar/perplexities/babyberta/optimal_ppl_v3')
    parser.add_argument('--grammar_dir', type=str, default='/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/reduce_grammar/perplexities')
    parser.add_argument('--top_k', type=float, default=0.2, choices=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument('--version', type=str, default='normal', choices=['normal', 'lexical', 'pos'],
                        help='Version of the corpus to evaluate.')
    parser.add_argument('--data_dir', type=str, default='/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/reduce_grammar/corpora')
    parser.add_argument('--size', type=int, default=None, help='Size of the dataset to evaluate on.')
    parser.add_argument('--device', default=None, choices=['cuda', 'cpu', 'mps'], help='Device to run the model on.')
    parser.add_argument('--optimal', action=argparse.BooleanOptionalAction)
    parser.add_argument('--hardware', type=str, default='local', choices=['local', 'snellius'])
    parser.add_argument('--max_parse_time', type=int, default=10)
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

    logger.info('Evaluating top_k: %s' % args.top_k)
    args = set_experiment_config(args)

    model, tokenizer = load_model_tokenizer(args)
    model.to(device)

    # size is here complete corpus size (not only test set)
    language, lm_tokenizer = load_language(args, encoder="transformer", corpus_size=args.size)
    add_special_token(language.grammar)
    prod2prob = create_prod2prob_dict(language.grammar)

    tokenized_dataset = [tokenizer(x)['input_ids'] for x in language.test_corpus]

    if args.model == 'babyberta':
        # perplexity methods for mlms
        ppl, all_token_probs, all_hidden_states = masked_ppl(
            model,
            tokenized_dataset,
            device,
            mask_token_id=tokenizer.mask_token_id, 
            skip_tokens={},
            batch_size=128,
            return_hidden=True,
        )

        store_model_probs(all_token_probs, language.test_corpus, f'{args.output_dir}/token_probs_eval_{args.model}_{args.version}_{args.top_k}.txt')
        bert_prob_dict = get_model_prob_dict(f'{args.output_dir}/token_probs_eval_{args.model}_{args.version}_{args.top_k}.txt')
        lm_probs, all_tokens = get_probs(language.test_corpus, tokenized_dataset, tokenizer, bert_prob_dict)

        with open(f'{args.output_dir}/token_probs_eval_{args.model}_{args.version}_{args.top_k}.pkl', 'wb') as f:
            pickle.dump((lm_probs, all_tokens), f)

    if args.optimal:
        avg_ppl, all_probs, num_parses, sen_lens, sen_ids, probs_per_word = pcfg_perplexity(
            language, 'all_parses', prod2prob, max_parse_time=args.max_parse_time, corpus_size=args.size, 
        )

        with open(f'{args.output_file_pcfg}/optimal_ppl_mlm_{args.version}_{args.top_k}_size_{args.size}_all_parses.pkl', 'wb') as f:
            pickle.dump((avg_ppl, all_probs, num_parses, sen_lens, sen_ids, probs_per_word), f)
    
    assert len(lm_probs) == len(probs_per_word), f'{len(lm_probs)} != {len(probs_per_word)}'

    print(f"LM-PPL {np.exp(-np.mean(lm_probs))}")
    print(f"PCFG-PPL {avg_ppl}")
    
    with open(f'{args.output_dir}/results_{args.model}_{args.version}_{args.top_k}_subset.json', 'w') as f:
        json.dump({
            'lm_ppl': np.exp(-np.mean(lm_probs)),
            'pcfg_ppl': avg_ppl,
            'spearman': spearmanr(lm_probs, probs_per_word)[0],
            'r2': r2_score(lm_probs, probs_per_word)
        }, f)

    del language
    del tokenizer
    del lm_tokenizer
    del model