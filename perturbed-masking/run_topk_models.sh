#!/bin/bash

# Exit immediately if any command exits with a non-zero status.
set -e

RESULTS_DIR=/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/perturbed-masking/test_results_v2
MODEL_DIR=/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/retrain/checkpoints_v2/
DATA_DIR=/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/reduce_grammar/corpora/
GRAMMAR_DIR=/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/reduce_grammar/grammars/nltk

versions=("normal")
topks=("0.5")
for topk in "${topks[@]}"; do
    for version in "${versions[@]}"; do
        echo "Running topk: $topk, version: $version"
        python main.py --metric dist \
                        --remove_punct \
                        --device cpu \
                        --model deberta \
                        --top_k "$topk" \
                        --embedding_layer \
                        --version "$version" \
                        --data  ${DATA_DIR} \
                        --grammar_path ${GRAMMAR_DIR} \
                        --home_model_path  ${MODEL_DIR} \
                        --output_dir ${RESULTS_DIR}/i_matrices/ \
                        --tree_path ${RESULTS_DIR}/ \
                        --eval_results_dir ${RESULTS_DIR}/eval/ \
                        --evaluation spearman \
                        --all_layers \
                        --split 1000
    done
done


