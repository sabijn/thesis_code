"""
Representations Extractor for ``transformers`` toolkit models.
Based on: https://github.com/davidarps/NeuroX/blob/4124a6996c1af948df3f470ca201ba7830e1c3d0/neurox/data/extraction/transformers_extractor.py 
"""

import argparse
import sys
import numpy as np
import torch
from tqdm import tqdm

def aggregate_repr(state, start, end, aggregation):
    """
    Function that aggregates activations/embeddings over a span of subword tokens.

    Returns a zero vector if end is less than start, i.e. the request is to
    aggregate over an empty slice.

    Parameters
    ----------
    state : numpy.ndarray
        Matrix of size [ NUM_LAYERS x NUM_SUBWORD_TOKENS_IN_SENT x LAYER_DIM]
    start : int
        Index of the first subword of the word being processed
    end : int
        Index of the last subword of the word being processed
    aggregation : {'first', 'last', 'average'}
        Aggregation method for combining subword activations

    Returns
    -------
    word_vector : numpy.ndarray
        Matrix of size [NUM_LAYERS x LAYER_DIM]
    """
    if end < start:
        sys.stderr.write("WARNING: An empty slice of tokens was encountered. " +
            "This probably implies a special unicode character or text " +
            "encoding issue in your original data that was dropped by the " +
            "transformer model's tokenizer.\n")
        return np.zeros((state.shape[0], state.shape[2]))
    if aggregation == "first":
        return state[:, start, :]
    elif aggregation == "last":
        return state[:, end, :]
    elif aggregation == "average":
        return np.average(state[:, start : end + 1, :], axis=1)


def extract_sentence_representations(sentence, model, tokenizer, tokenization_counts, device, dtype):
    """
    Get representations for one sentence
    """
    special_tokens = [
        x for x in tokenizer.all_special_tokens if x != tokenizer.unk_token
    ]

    special_tokens_ids = tokenizer.convert_tokens_to_ids(special_tokens)

    original_tokens = sentence.split(" ")
    # Add a letter and space before each word since some tokenizers are space sensitive
    tmp_tokens = [
        "a" + " " + x if x_idx != 0 else x for x_idx, x in enumerate(original_tokens)
    ]
    assert len(original_tokens) == len(tmp_tokens)

    with torch.no_grad():
        # Get tokenization counts if not already available
        for token_idx, token in enumerate(tmp_tokens):
            tok_ids = [
                x for x in tokenizer.encode(token) if x not in special_tokens_ids
            ]
            if token_idx != 0:
                # Ignore the first token (added letter)
                tok_ids = tok_ids[1:]

            if token in tokenization_counts:
                assert tokenization_counts[token] == len(tok_ids), "Got different tokenization for already processed word"
            else:
                tokenization_counts[token] = len(tok_ids)
        
        ids = tokenizer.encode(sentence, truncation=True)
        input_ids = torch.tensor([ids]).to(device)

        # Hugging Face format: tuple of torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)
        # Tuple has 13 elements for base model: embedding outputs + hidden states at each layer
        all_hidden_states = model(input_ids)[-1]
        all_hidden_states = [hidden_states[0].cpu().numpy() for hidden_states in all_hidden_states]
        all_hidden_states = np.array(all_hidden_states, dtype=dtype)

    # Remove special tokens
    ids_without_special_tokens = [x for x in ids if x not in special_tokens_ids]
    idx_without_special_tokens = [t_i for t_i, x in enumerate(ids) if x not in special_tokens_ids]

    filtered_ids = [ids[t_i] for t_i in idx_without_special_tokens]
    assert all_hidden_states.shape[1] == len(ids)
    all_hidden_states = all_hidden_states[:, idx_without_special_tokens, :]
    assert all_hidden_states.shape[1] == len(filtered_ids)

    segmented_tokens = tokenizer.convert_ids_to_tokens(filtered_ids)

    # Perform actual subword aggregation/detokenization
    counter = 0
    detokenized = []
    final_hidden_states = np.zeros((all_hidden_states.shape[0], len(original_tokens), all_hidden_states.shape[2]), dtype=dtype)
    inputs_truncated = False

    for token_idx, token in enumerate(tmp_tokens):
        current_word_start_idx = counter
        current_word_end_idx = counter + tokenization_counts[token]

        # Check for truncated hidden states in the case where the
        # original word was actually tokenized
        if  (tokenization_counts[token] != 0 and current_word_start_idx >= all_hidden_states.shape[1]) \
                or current_word_end_idx > all_hidden_states.shape[1]:
            final_hidden_states = final_hidden_states[:, :len(detokenized), :]
            inputs_truncated = True
            break

        final_hidden_states[:, len(detokenized), :] = aggregate_repr(
            all_hidden_states,
            current_word_start_idx,
            current_word_end_idx - 1,
            aggregation="average") # hardcoded!
        
        detokenized.append(
            "".join(segmented_tokens[current_word_start_idx:current_word_end_idx])
        )
        counter += tokenization_counts[token]

    if inputs_truncated:
        print("WARNING: Input truncated because of length, skipping check")
    else:
        assert counter == len(ids_without_special_tokens)
        assert len(detokenized) == len(original_tokens)

    return final_hidden_states, detokenized


def extract_representations(model, tokenizer, data_dir : str, device : str = "cpu", dtype : float = "float32", concat=True):
    print("Reading input corpus")
    def corpus_generator(input_corpus_path):
        with open(input_corpus_path, "r") as fp:
            for line in fp:
                yield line.strip()
            return
    
    tokenization_counts = {}
    print("Extracting representations from model")
    if concat: 
        return torch.vstack([extract_sentence_representations(sentence, model, tokenizer, tokenization_counts, device, dtype) 
                            for sentence_idx, sentence in enumerate(corpus_generator(data_dir))])
    else:
        return [extract_sentence_representations(sentence, model, tokenizer, tokenization_counts, device, dtype) 
                            for sentence_idx, sentence in enumerate(corpus_generator(data_dir))]