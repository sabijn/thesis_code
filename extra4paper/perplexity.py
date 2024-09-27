from tqdm import *
import numpy as np
import torch

from nltk import PCFG as nltk_PCFG, Production, ProbabilisticProduction
from nltk.parse import IncrementalLeftCornerChartParser as Parser

from tqdm import tqdm
import numpy as np
import signal
import time
import torch


def masked_ppl(model, corpus, device, mask_token_id=None, skip_tokens=None, batch_size=128, return_hidden=False):
    if skip_tokens is None:
        skip_tokens = set()

    model.eval()

    total_tokens = sum(map(len, corpus))
    max_sen_len = max(map(len, corpus))

    # Set all masked tokens in single tensor.
    all_token_ids = torch.zeros(total_tokens, max_sen_len).to(device).int()
    all_attention_masks = torch.zeros(total_tokens, max_sen_len).to(device).int()
    mask_positions = []
    flat_token_ids = []    

    current_idx = 0

    for input_ids in corpus:
        sen_len = len(input_ids)
        current_slice = slice(current_idx, current_idx+sen_len)
        repeated_input_ids = torch.tensor(input_ids).repeat(sen_len, 1).to(device)
        all_token_ids[current_slice, :sen_len] = repeated_input_ids
        all_token_ids[current_slice, :sen_len].fill_diagonal_(mask_token_id)
        
        all_attention_masks[current_slice, :sen_len] = 1
        
        mask_positions.extend(range(sen_len))
        flat_token_ids.extend(input_ids)

        current_idx += sen_len

    mask_positions = torch.tensor(mask_positions).to(device)
    flat_token_ids = torch.tensor(flat_token_ids).to(device)

    # Create model logits for all masked tokens.
    # Torch.split (chunk the input tensor (0) in sizes (1))
    input_iterator = torch.split(all_token_ids, batch_size)
    attention_iterator = torch.split(all_attention_masks, batch_size)
    mask_positions_iterator = torch.split(mask_positions, batch_size)
    flat_token_ids_iterator = torch.split(flat_token_ids, batch_size)

    iterator = zip(
        input_iterator, 
        attention_iterator, 
        mask_positions_iterator, 
        flat_token_ids_iterator
    )

    all_token_probs = []
    all_hidden_states = {layer_idx: [] for layer_idx in range(model.config.num_hidden_layers+1)}

    for idx, batch in enumerate(tqdm(iterator, total=len(input_iterator))):
        input_ids_batch, attention_mask_batch, mask_positions, flat_tokens = batch
        with torch.no_grad():
            batch_slice = slice(idx*batch_size, (idx+1)*batch_size)
            batch_output = model(
                input_ids=input_ids_batch, 
                attention_mask=attention_mask_batch,
                output_hidden_states=return_hidden,
            )
            batch_logits = batch_output.logits
            batch_probs = batch_logits.log_softmax(-1)

        current_idx = 0

        current_bsz = batch_probs.shape[0]
        
        mask_probs = batch_probs[range(current_bsz), mask_positions]
        token_probs = mask_probs[range(current_bsz), flat_tokens]
        if return_hidden:
            for layer_idx, hidden_states in enumerate(batch_output.hidden_states):
                mask_states = hidden_states[range(current_bsz), mask_positions]
                unk_mask = [token_id.item() not in skip_tokens for token_id in flat_tokens]
                all_hidden_states[layer_idx].append(mask_states[unk_mask])
        
        for prob, token_id in zip(token_probs, flat_tokens):
            if token_id.item() not in skip_tokens:
                all_token_probs.append(prob.item())

    ppl = np.exp(-np.sum(all_token_probs)/len(all_token_probs))
    
    if return_hidden:
        return ppl, all_token_probs, all_hidden_states
    else:
        return ppl, all_token_probs


def causal_ppl(model, corpus, device, skip_tokens):
    model.eval()
    
    all_token_probs = []
    all_sen_probs = []

    for input_ids_list, sen in tqdm(zip(corpus['input_ids'], corpus['text'])):
        input_ids = torch.tensor(input_ids_list).unsqueeze(0).to(device)
        sen_len = input_ids.shape[-1]
        sen_probs = []

        with torch.no_grad():
            probs = model(input_ids[:, :-1]).logits.log_softmax(-1)[0]
            
        for idx, prob_row in enumerate(probs, start=1):
            token_id = input_ids[0, idx].item()
            
            if token_id not in skip_tokens:
                token_prob = prob_row[token_id].item()
                all_token_probs.append(token_prob)
                sen_probs.append(token_prob)

        all_sen_probs.append(sen_probs)

    ppl = np.exp(-np.mean(all_token_probs))
    
    return ppl, all_token_probs, all_sen_probs


class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException

signal.signal(signal.SIGALRM, timeout_handler)


def pcfg_perplexity(lm_language, method, prod2prob, max_parse_time=10, corpus_size=None, sen_ids_filter=None, verbose=False):
    all_probs = []
    sen_lens = []
    num_parses = []
    sen_ids = []
    probs_per_word = []
    
    chart_parser = Parser(lm_language.grammar)
    corpus = lm_language.test_corpus
    iterator = tqdm(corpus) if verbose else corpus

    # For every sentence in the corpus
    for sen_idx, sen in enumerate(iterator):
        if sen_ids_filter is not None and sen_idx not in sen_ids_filter:
            continue

        orig_tree = lm_language.tree_corpus[sen]
        sen = sen.split()
        sen_len = len(sen)

        if method == 'all_parses':
            weighted_leaf_probs = []
            num_sen_parses = []
            skip = False

            signal.alarm(max_parse_time)
            try:
                # For every leaf in the sentence
                for idx, orig_leaf in enumerate(sen):
                    sen2 = list(sen)
                    # Replace the leaf with '<X>'
                    sen2[idx] = '<X>'

                    tree_probs = []
                    leaf_probs = []

                    # For every possible part of the sentence
                    for i, tree in enumerate(chart_parser.parse(sen2)):
                        # Get the product of all productions in the current tree
                        tree_prob = np.prod([(prod2prob[prod]) for prod in tree.productions()])

                        # Get the production of the currently masked token (terminal)
                        leaf_idx_prod = [prod for prod in tree.productions() if isinstance(prod.rhs()[0], str)][idx]
                        # Get the index of the non-terminal
                        leaf_idx_pos = leaf_idx_prod.lhs()
                        # Get the probability of the currently masked token
                        orig_leaf_prob = prod2prob[Production(leaf_idx_pos, (orig_leaf,))]

                        tree_probs.append(tree_prob)
                        leaf_probs.append(orig_leaf_prob)

                    num_sen_parses.append(i+1)
                    tree_probs_sum = np.sum(tree_probs)

                    # Calculate the weighted probability of the masked token
                    # (tree prob times leaf prob) / sum of tree probs (marginalize, even general formule opzoeken)
                    weighted_leaf_prob = sum((tree_prob/tree_probs_sum) * leaf_prob for tree_prob, leaf_prob in zip(tree_probs, leaf_probs))
                    weighted_leaf_probs.append(np.log(weighted_leaf_prob))
            except TimeoutException:
                continue
            finally:
                signal.alarm(0)

            sen_ids.append(sen_idx)
            num_parses.append(num_sen_parses)
            all_probs.append(np.sum(weighted_leaf_probs))  
            probs_per_word.extend(weighted_leaf_probs)
                  
        elif method == 'sen_parses':
            sen_leaf_probs = []
            sen_tree_probs = []

            start_time = time.time()
            signal.alarm(max_parse_time)
            try:
                parses = list(chart_parser.parse(sen))
            except TimeoutException:
                continue
            finally:
                signal.alarm(0)

            for i, tree in enumerate(parses):
                leaf_probs = [prod2prob[prod] for prod in tree.productions() if isinstance(prod.rhs()[0], str)]
                leaf_prob = np.prod(leaf_probs)
                tree_prob = np.prod([(prod2prob[prod]) for prod in tree.productions()])# if not isinstance(prod.rhs()[0], str)])

                sen_leaf_probs.append(leaf_prob)
                sen_tree_probs.append(tree_prob)

            total_sen_tree_probs = sum(sen_tree_probs)
            weighted_sen_prob = sum(
                (tree_prob/total_sen_tree_probs) * leaf_prob 
                for tree_prob, leaf_prob in zip(sen_tree_probs, sen_leaf_probs)
            )
            weighted_sen_logprob = np.log(weighted_sen_prob)

            sen_ids.append(sen_idx)
            num_parses.append(i+1)
            all_probs.append(weighted_sen_logprob)
        elif method == 'current_parse':
            leaf_prods = [prod for prod in orig_tree.productions() if isinstance(prod.rhs()[0], str)]
            word_probs = [prod2prob[prod] for prod in leaf_prods]
            sen_prob = np.sum(word_probs)
            
            sen_ids.append(sen_idx)
            all_probs.append(sen_prob)
            probs_per_word.extend(word_probs)
        else:
            raise ValueError(method)

        sen_lens.append(sen_len)
                
    avg_ppl = np.exp(-np.sum(all_probs)/np.sum(sen_lens))
    
    return avg_ppl, all_probs, num_parses, sen_lens, sen_ids, probs_per_word