#!/bin/bash

# Exit immediately if any command exits with a non-zero status.
set -e

# Set the path to the edge probing repository.
MODEL=babyberta
EDGE_PROBING_DIR=/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/edge-probing
DATA_DIR=/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/reduce_grammar/corpora
MODEL_DIR=/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/retrain/checkpoints/${MODEL}
OUTPUT_DIR=/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/constprobing-repr/data

topks=("0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9")
versions=("normal")
for version in "${versions[@]}"; do
    for topk in "${topks[@]}"; do
        echo "Creating activations for topk=${topk} and version=${version}"
        python create_activations.py --model_type ${MODEL} \
            --version ${version} \
            --top_k ${topk} \
            --data_dir ${DATA_DIR}/${version}/test_trees_${version}_${topk}.txt \
            --output_dir ${OUTPUT_DIR}/${MODEL}/${version}/${topk} \
            --concat
    done
done