#!/bin/bash

# Exit immediately if any command exits with a non-zero status.
set -e

# Set working dir
cd /Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/retrain

# Run the script
echo 
python eval.py --model babyberta \
                --model_dir checkpoints \
                --data_dir /Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/reduce_grammar/corpora \
                --output_dir /Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/retrain/results \
                --top_k 0.2 \
                --version normal \
                --size 10 \
                --device cpu