from transformers import AutoModelForMaskedLM, AutoModelForCausalLM
from transformers import PreTrainedTokenizer, BertTokenizer
import json
import os

class CustomTokenizer(PreTrainedTokenizer):
    def __len__(self):
        return len(self.vocab)

    @property
    def vocab_size(self):
        return len(self.vocab)
    
    def save_vocabulary(self, *args, **kwargs):
        return BertTokenizer.save_vocabulary(self, *args, **kwargs)
    
    def _tokenize(self, sen: str):
        return sen.split(" ")
    
    def _convert_token_to_id(self, w: str):
        return self.vocab.get(w, self.vocab[self.unk_token])

def create_tf_tokenizer_from_vocab(
    vocab, 
    unk_token: str = '<unk>', 
    pad_token: str = '<pad>',
    mask_token: str = '<mask>',
):
    tokenizer = CustomTokenizer()

    tokenizer.added_tokens_encoder = vocab
    tokenizer.added_tokens_decoder = {idx: w for w, idx in vocab.items()}
    tokenizer.vocab = tokenizer.added_tokens_encoder
    tokenizer.ids_to_tokens = tokenizer.added_tokens_decoder
    
    tokenizer.unk_token = unk_token
    tokenizer.pad_token = pad_token
    tokenizer.mask_token = mask_token

    return tokenizer

def load_model(checkpoint, device):
    model = AutoModelForMaskedLM.from_pretrained(checkpoint).to(device)

    with open(f'{checkpoint}/added_tokens.json') as f:
        vocab = json.load(f)
    tokenizer = create_tf_tokenizer_from_vocab(vocab)

    return model, tokenizer

def load_model_tokenizer(args):
    if args.model_type == 'gpt2':
        automodel = AutoModelForCausalLM
    elif args.model_type == 'babyberta':
        automodel = AutoModelForMaskedLM
    else:
        raise ValueError(f'Model {args.model} not supported.')
    
    model = automodel.from_pretrained(args.model_file)

    with open(f'{args.model_file}added_tokens.json') as f:
        vocab = json.load(f)

    tokenizer = create_tf_tokenizer_from_vocab(vocab)

    return model, tokenizer


def set_experiment_config(args):
    # Check if model directory exists and find the correct checkpoint
    if not os.path.exists(args.model_path):
        raise ValueError(f'Model directory {args.model_path} does not exist.')
    
    highest_config = 0
    print(args.model_path)
    for dir_name in os.listdir(args.model_path):
        if dir_name.split('-')[0] == 'checkpoint':
            config = int(dir_name.split('-')[1])
            if config > highest_config:
                highest_config = config

    args.model_file = f'{args.model_path}/checkpoint-{highest_config}/'

    return args
