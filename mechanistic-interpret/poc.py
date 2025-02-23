import argparse
from pathlib import Path
import torch.cuda
import torch.nn.functional as F

from utils import load_model

def get_model_clm_probs(model, tokenizer):
    model.eval()
    # I needed the bed from the door .
    sent = "I needed the bed from the"

    input_ids = tokenizer(sent, return_tensors="pt").input_ids
    
    layers = model.transformer.h

    # Store logits per layer
    probs_layers = []

    # Pass through each layer and collect logits
    with torch.no_grad():
        # Initial hidden state from the embeddings
        hidden_state = model.transformer.wte(input_ids)
        
        for layer_idx, layer in enumerate(layers):
            # Process each layer output
            hidden_state = layer(hidden_state)[0]
            layer_output = model.lm_head(model.transformer.ln_f(hidden_state))

            # Apply softmax to obtain probabilities and save the result
            probs = F.softmax(layer_output, dim=-1)
            probs_layers.append(probs)

    probs = torch.cat(probs_layers, dim=0)
    max_probs, tokens = probs.max(dim=-1) # dit eruit en pak de positie van het woord dat je wil hebben

    # Decode token IDs to words for each layer
    words = [[tokenizer.decode(t) for t in layer_tokens] for layer_tokens in tokens]
    
    # Decode the input prompt tokens
    input_words = [tokenizer.decode(t) for t in input_ids]

    # Print the input words and the decoded words per layer
    print("Input words:", input_words)
    print("Words predicted for the masked token per layer:")
    for idx, layer_words in enumerate(words):
        print(f"Layer {idx + 1}: {layer_words}")

    return max_probs, tokens

def get_model_mlm_probs(model, tokenizer):
    # Prepare the prompt and tokenize it
    sent = "Finally he stared across the screen to keep The <mask> <apostrophe>s bed to my heat heaven outside ."
    label = "crater"
    label_id = tokenizer.convert_tokens_to_ids(label)

    input = tokenizer(sent, return_tensors="pt")
    
    input_ids = input["input_ids"]
    attention_mask = input["attention_mask"]

    # Identify the position of the masked token
    mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]

    with torch.no_grad():
        outputs = model.deberta(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

    # Extract the hidden states from the output
    hidden_states = outputs.hidden_states  # This contains the hidden states for each layer

    # Store logits per layer
    probs_layers = []

    # Iterate over the hidden states of each layer
    for layer_idx, hidden_state in enumerate(hidden_states):
        # Apply the final layer normalization and LM head to get predictions
        layer_output = model.cls.predictions(hidden_state)

        # Apply softmax only to the masked token position to get probabilities
        mask_logits = layer_output[:, mask_token_index, :]
        probs = F.softmax(mask_logits, dim=-1)
        probs_layers.append(probs)

    # Concatenate probabilities from all layers
    probs = torch.cat(probs_layers, dim=0)

    # Find the maximum probability and corresponding tokens for the masked position
    print(probs.shape)
    max_probs, tokens = probs.max(dim=-1)
    good_probs = probs[:, :, label_id]
    print(good_probs)
    print(max_probs)

    # Decode token IDs to words for each layer
    words = [[tokenizer.decode(t) for t in layer_tokens] for layer_tokens in tokens]

    # Decode the input prompt tokens
    input_words = [tokenizer.decode(t) for t in input_ids]

    # Print the input words and the decoded words per layer
    print("Input words:", input_words)
    print("Words predicted for the masked token per layer:")
    for idx, layer_words in enumerate(words):
        print(f"Layer {idx + 1}: {layer_words}")

    return max_probs, tokens

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mechanistic Interpretation')
    parser.add_argument('--model_dir', type=Path, default='/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/pcfg-lm/resources/checkpoints/')
    parser.add_argument('--model', type=str, default='deberta', choices=['deberta', 'gpt2'])
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    model_dir = args.model_dir / args.model

    if args.device is None:
        if torch.cuda.is_available():
            # For running on snellius
            device = torch.device("cuda")
        else:
            # For running on laptop
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    model, tokenizer = load_model(model_dir, device, args.model)
    
    if args.model == 'gpt2':
        max_probs, tokens = get_model_clm_probs(model, tokenizer)
    elif args.model == 'deberta':
        max_probs, tokens = get_model_mlm_probs(model, tokenizer)
    else:
        raise ValueError(f'Model {args.model} not supported.')

    # basseer op loop through retrain/eval.py