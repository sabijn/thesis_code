#!/bin/bash

# Exit immediately if any command exits with a non-zero status.
set -e

# Set the path to the edge probing repository.
EDGE_PROBING_DIR=/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/edge-probing
MODEL=babyberta
DATA_DIR=/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/reduce_grammar/

topks=("0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9")
versions=("normal" "lexical")
for topk in "${topks[@]}"; do
    for version in "${versions[@]}"; do
        echo "Running edge probing for topk=${topk} and version=${version}"
        python main.py --version ${version} \
                          --topk ${topk} \
                          --model ${MODEL} \
                          --created_states_path ${EDGE_PROBING_DIR}/data/${MODEL}/${version}/${topk} \
                          --data 


                          --output_dir ${EDGE_PROBING_DIR}/output \
                          --data_dir ${EDGE_PROBING_DIR}/data \
                          --model_dir ${EDGE_PROBING_DIR}/models