import nltk
from transformers import PreTrainedModel, AutoModelForMaskedLM
from tqdm import *
from tokenizer import *
import argparse
import torch
from pathlib import Path
import random
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import pickle
import os


def load_model(checkpoint, device):
    model = AutoModelForMaskedLM.from_pretrained(checkpoint).to(device)

    with open(f'{checkpoint}/added_tokens.json') as f:
        vocab = json.load(f)
    tokenizer = create_tf_tokenizer_from_vocab(vocab)

    return model, tokenizer

def tree_to_pos(tree, skip_unk_tokens=False):
    pos_tags = [
        prod.lhs().symbol().split("_")[0]
        for prod in tree.productions()
        if isinstance(prod.rhs()[0], str)
    ]
    assert len(pos_tags) == len(tree.leaves())
    if skip_unk_tokens:
        no_unk_pos = []
        for pos, w in zip(pos_tags, tree.leaves()):
            if w in tokenizer.vocab:
                no_unk_pos.append(pos)
        return no_unk_pos
    else:
        return pos_tags

def create_states(
    tokenizer, 
    tree_corpus, 
    model, 
    device,
    concat=True, 
    skip_cls=False, 
    num_items=None,
    verbose=False,
    all_layers=True,
    skip_unk_tokens=False,
):
    if isinstance(model, PreTrainedModel):
        all_sens = [torch.tensor(tokenizer.convert_tokens_to_ids(tree.leaves())) for tree in tree_corpus]
        pad_idx = tokenizer.pad_token_id
        num_parameters = model.num_parameters()
    else:
        all_sens = [tokenizer.tokenize(tree.leaves(), pos_tags=tree_to_pos(tree)) for tree in tree_corpus]
        pad_idx = tokenizer.pad_idx
        num_parameters = model.num_parameters

    if num_items is not None:
        all_sens = random.sample(all_sens, num_items)
    print(len(all_sens))
    lengths = [len(sen) for sen in all_sens]
    sen_tensor = pad_sequence(all_sens, padding_value=pad_idx, batch_first=True).to(device)

    batch_size = int(1e9 / num_parameters)
    print(batch_size)
    states = defaultdict(list) if all_layers else []
    iterator = range(0, len(all_sens), batch_size)
    if verbose:
        iterator = tqdm(iterator)

    for idx in iterator:
        batch = sen_tensor[idx: idx + batch_size]

        with torch.no_grad():
            all_hidden = model(batch, output_hidden_states=True).hidden_states

        if all_layers:
            for layer_idx, layer_hidden in enumerate(all_hidden):
                for hidden, sen, length in zip(layer_hidden, batch, lengths[idx: idx + batch_size]):
                    unk_mask = sen[:length] != tokenizer.unk_token_id
                    states[layer_idx].append(hidden[:length][unk_mask])
        else:
            states.extend([
                hidden[int(skip_cls):length]
                for hidden, length in zip(all_hidden[-1], lengths[idx: idx + batch_size])
            ])

    if concat:
        if all_layers:
            for layer_idx, layer_states in states.items():
                states[layer_idx] = torch.concat(layer_states)
            return states
        else:
            return torch.concat(states)
    else:
        return states

if __name__ == '__main__':
    """
    Run with: python create_activations.py --checkpoint deberta
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', required=True, choices=['deberta', 'gpt2', 'babyberta'])
    parser.add_argument('--version', default='normal')
    parser.add_argument('--top_k', default=1.0)
    parsedargs = parser.parse_args()

    if parsedargs.top_k == 1.0:
        model_path = Path(f'pcfg-lm/resources/checkpoints/{parsedargs.model_type}/')
        with open('corpora/eval_trees_10k.txt') as f:
            tree_corpus = [nltk.Tree.fromstring(l.strip()) for l in f]
        
        output_dir = Path('data')
    else:
        model_path = Path(f'/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/retrain/checkpoints/{parsedargs.model_type}/{parsedargs.version}/{parsedargs.top_k}/')

        # Check if model directory exists and find the correct checkpoint
        if not os.path.exists(model_path):
            raise ValueError(f'Model directory {model_path} does not exist.')
        
        highest_config = 0
        for dir_name in os.listdir(model_path):
            if dir_name.split('-')[0] == 'checkpoint':
                config = int(dir_name.split('-')[1])
                if config > highest_config:
                    highest_config = config

        model_path = f'{model_path}/checkpoint-{highest_config}/'

        # Load data
        with open(f'/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/reduce_grammar/corpora/{parsedargs.version}/all_trees_{parsedargs.version}_{parsedargs.top_k}.txt') as f:
            trees = [l.strip() for l in f][990_000:]
            tree_corpus = [nltk.Tree.fromstring(tree) for tree in trees]
        
        output_dir = Path(f'data/{parsedargs.version}/{parsedargs.top_k}')

        output_dir.mkdir(parents=True, exist_ok=True)


    if torch.cuda.is_available():
        # For running on snellius
        device = torch.device("cuda")
        print('Running on GPU.')
    # elif torch.backends.mps.is_available():
    #     # For running on M1
    #     device = torch.device("mps")
    #     print('Running on M1 GPU.')
    else:
        # For running on laptop
        device = torch.device("cpu")
        print('Running on CPU.')

    all_test_mccs = []

    # Load model
    model, tokenizer = load_model(model_path, device)
    model.eval()

    # extract hidden states from the model
    all_layer_states = create_states(
        tokenizer, 
        tree_corpus, 
        model, 
        device,
        concat=False, 
        skip_cls=False, 
        verbose=True, 
        all_layers=True, 
        skip_unk_tokens=True
    )

    print(all_layer_states.keys())
    # store all_layer_states in pickle
    with open(output_dir / 'activations_notconcat.pickle', 'wb') as f:
        pickle.dump(all_layer_states, f)

    # for layer_idx, states in tqdm(all_layer_states.items()):
    #     if layer_idx < 8:
    #         continue
    #     with torch.no_grad():
    #         states = model.cls.predictions.transform(states)
            
    #     dc, test_mcc, _, _ = train_dc(
    #         tree_corpus, 
    #         states, 
    #         verbose=True, 
    #         train_epochs=5, 
    #         train_ids=train_ids,
    #         dev_ids=dev_ids,
    #         test_ids=test_ids,
    #         separate_tokens=True,
    #         skip_unk_tokens=True,
    #     )
    #     test_mccs.append(test_mcc)

    # all_test_mccs.append(test_mccs)