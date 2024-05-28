#!/bin/bash

# Exit immediately if any command exits with a non-zero status.
set -e

DATA_DIR=/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/constprobing-repr/data
MODEL=babyberta
topks=("0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9")
versions=("normal")
for version in "${versions[@]}"; do
    for topk in "${topks[@]}"; do
        echo "Combining activations for topk=${topk} and version=${version}"
        current_data_dir=$DATA_DIR/$MODEL/$version/$topk
        python clean_gold_trees.py --data_dir $current_data_dir
    done
done