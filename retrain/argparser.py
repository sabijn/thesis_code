import argparse
from pathlib import Path
import logging 
import torch

logger = logging.getLogger(__name__)

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--data_dir', type=str, 
                        default='/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/reduce_grammar/corpora/results')
    parser.add_argument('--version', type=str, default='normal', choices=['pos', 'lexical', 'normal'])
    parser.add_argument('--top_k', type=float, default=0.2)
    
    parser.add_argument('--base_model', type=str, default='phueb/BabyBERTa-1', choices=['phueb/BabyBERTa-1', 'distilgpt2'])
    parser.add_argument('--save_steps', type=int, default=10_000)
    parser.add_argument('--eval_steps', type=int, default=100)
    parser.add_argument('--logging_steps', type=int, default=100)
    parser.add_argument('--per_device_train_batch_size', type=int, default=64)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=64)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--lr_scheduler_type', type=str, default='cosine')
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--fp16', action='store_false')
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--group_by_length', action='store_false')
    parser.add_argument('--auto_find_batch_size', action='store_true')
    parser.add_argument('--do_eval', action='store_false')
    parser.add_argument('--evaluation_strategy', type=str, default='steps', choices=['steps', 'epoch'])
    parser.add_argument('--epochs', type=int, default=1)
    
    args = parser.parse_args()

    return args