#!/bin/bash

set -e

echo "Calculating optimal and model perplexity for babyberta"
python get_model_probs.py --model babyberta \
                --version normal \
                --top_k "0.2" \
                --device cpu \
                --size 100 \
                --optimal \
                --max_parse_time 10 \
                --output_dir /Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/extra4paper/results \
                --model_dir /Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/retrain/checkpoints_v2 \
                --mutated_dataset test_mutations.txt
