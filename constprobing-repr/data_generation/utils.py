from transformers import AutoModelForMaskedLM
from transformers import PreTrainedTokenizer, BertTokenizer
import json

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