import argparse
from pathlib import Path
import torch.cuda
import torch.nn.functional as F

from utils import load_model, store_model_probs
from data import load_eval_data
from model_probs import get_model_mlm_bigram_probs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mechanistic Interpretation')
    parser.add_argument('--model_dir', type=Path, default='/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/pcfg-lm/resources/checkpoints/')
    parser.add_argument('--model', type=str, default='deberta', choices=['deberta', 'gpt2'])
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--top_k', type=float, default=0.2, choices=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    parser.add_argument('--data_file', type=str, default='/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/reduce_grammar/corpora/test_sent_normal_0.2.txt')
    parser.add_argument('--output_dir', type=Path, default='/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/retrain/results')
    parser.add_argument('--size', type=int, default=None, help='Size of the dataset to evaluate on.')
    parser.add_argument('--log_probs', type=str, default='False', choices=['True', 'False'])

    args = parser.parse_args()
    args.log_probs = bool(eval(args.log_probs))

    model_dir = args.model_dir / args.model
    if args.log_probs:
        output_dir = args.output_dir / args.model / str(args.top_k) / 'log_probs'
    else:
        output_dir = args.output_dir / args.model / str(args.top_k) / 'probs'
    output_dir.mkdir(parents=True, exist_ok=True)

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
    model.to(device)

    dataset = load_eval_data(args, tokenizer, args.data_file, test_size=args.size)
    
    if args.model == 'gpt2':
        raise ValueError(f'Bigram not implemented for clm')
        
    elif args.model == 'deberta':
        all_token_probs = get_model_mlm_bigram_probs(model, 
                                                dataset['test']['input_ids'],
                                                device,
                                                mask_token_id=tokenizer.mask_token_id,
                                                skip_tokens={},
                                                batch_size=128,
                                                return_hidden=True,
                                                log_probs=args.log_probs)
    else:
        raise ValueError(f'Model {args.model} not supported.')

    for (idx, value) in all_token_probs.items():
        store_model_probs(value, dataset, output_dir / f'bigram_probs_eval_{args.model}_{args.top_k}_layer_{idx}.txt')

