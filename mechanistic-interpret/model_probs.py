import torch
import numpy as np
from tqdm import tqdm

def get_model_mlm_probs(model, corpus, device, mask_token_id=None, skip_tokens=None, batch_size=128, return_hidden=True, log_probs=True):
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

    all_layer_probs = {layer_idx: [] for layer_idx in range(model.config.num_hidden_layers+1)}

    for _, batch in enumerate(tqdm(iterator, total=len(input_iterator))):
        input_ids_batch, attention_mask_batch, mask_positions, flat_tokens = batch
        with torch.no_grad():
            batch_hidden_states = model.deberta(
                input_ids=input_ids_batch, 
                attention_mask=attention_mask_batch,
                output_hidden_states=return_hidden,
            ).hidden_states

            current_bsz = batch_hidden_states[0].shape[0]

            for layer_idx, hidden_states in enumerate(batch_hidden_states):
                layer_output = model.cls.predictions(hidden_states)
                mask_states = layer_output[range(current_bsz), mask_positions]
                mask_probs = mask_states.log_softmax(-1) if log_probs else mask_states.softmax(-1)
                unk_mask = [token_id.item() not in skip_tokens for token_id in flat_tokens]
                all_layer_probs[layer_idx].extend(mask_probs[unk_mask][range(current_bsz), flat_tokens].tolist())
    
    return all_layer_probs

def get_model_clm_probs(model, corpus, device, skip_tokens, log_probs=True):
    model.eval()
    
    all_layer_probs = {layer_idx: [] for layer_idx in range(model.config.num_hidden_layers+1)}
    layers = model.transformer.h

    for input_ids_list in tqdm(corpus['input_ids']):
        input_ids = torch.tensor(input_ids_list).unsqueeze(0).to(device)

        with torch.no_grad():
            hidden_state = model.transformer.wte(input_ids)

            for layer_idx, layer in enumerate(layers):
                hidden_state = layer(hidden_state)[0]
                layer_output = model.lm_head(model.transformer.ln_f(hidden_state))[:, :-1]
                layer_output = layer_output.log_softmax(-1)[0] if log_probs else layer_output.softmax(-1)[0]

                for idx, prob_row in enumerate(layer_output, start=1):
                    token_id = input_ids[0, idx].item()
                    
                    if token_id not in skip_tokens:
                        token_prob = prob_row[token_id].item()
                        all_layer_probs[layer_idx].append(token_prob)
    
    return all_layer_probs

def get_model_mlm_bigram_probs(model, corpus, device, mask_token_id=None, skip_tokens=None, batch_size=128, return_hidden=True, log_probs=True):
    if skip_tokens is None:
        skip_tokens = set()

    model.eval()

    total_tokens = sum(map(len, corpus)) - len(corpus)  # Adjust for bigrams
    max_sen_len = max(map(len, corpus))

    # Set all masked tokens in a single tensor
    all_token_ids = torch.zeros(total_tokens, max_sen_len).to(device).int()
    all_attention_masks = torch.zeros(total_tokens, max_sen_len).to(device).int()
    bigram_positions = []
    flat_token_ids_pairs = []

    current_idx = 0

    for input_ids in corpus:
        sen_len = len(input_ids) - 1  # Adjust for bigrams within the sentence
        current_slice = slice(current_idx, current_idx + sen_len)

        # Create bigram pairs by masking two consecutive tokens
        for i in range(sen_len):
            repeated_input_ids = torch.tensor(input_ids).repeat(sen_len, 1).to(device)
            repeated_input_ids[i, i] = mask_token_id
            repeated_input_ids[i, i + 1] = mask_token_id
            
            all_token_ids[current_slice, :len(input_ids)] = repeated_input_ids
            all_attention_masks[current_slice, :len(input_ids)] = 1
            bigram_positions.append((i, i + 1))
            flat_token_ids_pairs.append((input_ids[i], input_ids[i + 1]))

        current_idx += sen_len

    # Convert lists to tensors
    bigram_positions = torch.tensor(bigram_positions).to(device)
    flat_token_ids_pairs = torch.tensor(flat_token_ids_pairs).to(device)

    # Create model logits for all masked tokens.
    input_iterator = torch.split(all_token_ids, batch_size)
    attention_iterator = torch.split(all_attention_masks, batch_size)
    bigram_positions_iterator = torch.split(bigram_positions, batch_size)
    flat_token_ids_pairs_iterator = torch.split(flat_token_ids_pairs, batch_size)

    iterator = zip(input_iterator, attention_iterator, bigram_positions_iterator, flat_token_ids_pairs_iterator)

    all_layer_probs = {layer_idx: [] for layer_idx in range(model.config.num_hidden_layers + 1)}

    for _, batch in enumerate(tqdm(iterator, total=len(input_iterator))):
        input_ids_batch, attention_mask_batch, bigram_positions_batch, flat_token_ids_pairs_batch = batch
        with torch.no_grad():
            batch_hidden_states = model.deberta(
                input_ids=input_ids_batch,
                attention_mask=attention_mask_batch,
                output_hidden_states=return_hidden,
            ).hidden_states

            current_bsz = batch_hidden_states[0].shape[0]

            for layer_idx, hidden_states in enumerate(batch_hidden_states):
                layer_output = model.cls.predictions(hidden_states)
                
                # Retrieve probabilities for both tokens in the bigram
                bigram_probs = []
                for i in range(current_bsz):
                    pos1, pos2 = bigram_positions_batch[i]
                    token1, token2 = flat_token_ids_pairs_batch[i]
                    
                    # Get probability for each token in the bigram
                    prob1 = layer_output[i, pos1].log_softmax(-1)[token1] if log_probs else layer_output[i, pos1].softmax(-1)[token1]
                    prob2 = layer_output[i, pos2].log_softmax(-1)[token2] if log_probs else layer_output[i, pos2].softmax(-1)[token2]
                    
                    bigram_prob = prob1 + prob2 if log_probs else prob1 * prob2  # Sum for log-prob, multiply for softmax
                    bigram_probs.append(bigram_prob.item())
                
                all_layer_probs[layer_idx].extend(bigram_probs)
    
    return all_layer_probs