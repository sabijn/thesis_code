#!/bin/bash

# Exit immediately if any command exits with a non-zero status.
set -e

# Set the path to the edge probing repository.
MODEL=babyberta
EDGE_PROBING_DIR=/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/edge-probing
DATA_DIR=/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/reduce_grammar/corpora
MODEL_DIR=/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/retrain/checkpoints/${MODEL}
OUTPUT_DIR=/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/edge-probing/results

topks=("0.6" "0.7" "0.8" "0.9")
versions=("normal" "lexical")
for version in "${versions[@]}"; do
    for topk in "${topks[@]}"; do
        echo "Running edge probing for topk=${topk} and version=${version}"
        python main.py --version ${version} \
                          --top_k ${topk} \
                          --model ${MODEL} \
                          --created_states_path ${EDGE_PROBING_DIR}/data/${MODEL}/${version}/${topk} \
                          --data ${DATA_DIR}/${version}/corpus_${topk}_${version}.pt \
                          --home_model_path ${MODEL_DIR}/${version}/${topk}/ \
                          --output_path ${OUTPUT_DIR}/${MODEL}/${version}/${topk} \
                          --span_ids_path ${EDGE_PROBING_DIR}/data/${MODEL}/${version}/${topk}/span_ids.pkl \
                          --tokenized_labels_path ${EDGE_PROBING_DIR}/data/${MODEL}/${version}/${topk}/tokenized_labels.pkl \
                          --test_data ${DATA_DIR}/${version}/test_trees_${version}_${topk}.txt \
                          --device cpu
    done
done