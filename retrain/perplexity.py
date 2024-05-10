from tqdm import *
import numpy as np
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


def causal_ppl(model, corpus, skip_tokens):
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