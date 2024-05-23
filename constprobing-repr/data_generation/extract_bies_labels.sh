#!/bin/bash

# Exit immediately if any command exits with a non-zero status.
set -e

# Set the path to the edge probing repository.
MODEL=babyberta
DATA_DIR=/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/reduce_grammar/corpora
MODEL_DIR=/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/retrain/checkpoints/${MODEL}
OUTPUT_DIR=/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/constprobing-repr/data

topks=("0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9")
versions=("normal")
for version in "${versions[@]}"; do
    for topk in "${topks[@]}"; do
        echo "Creating bies labels for topk=${topk} and version=${version}"
        python extract_bies_labels.py --model_type ${MODEL} \
            --version ${version} \
            --top_k ${topk} \
            --data_dir ${DATA_DIR}/${version}/test_trees_${version}_${topk}.txt \
            --text_toks ${OUTPUT_DIR}/${MODEL}/${version}/${topk}/train_text_bies.txt \
            --bies_labels ${OUTPUT_DIR}/${MODEL}/${version}/${topk}/train_bies_labels.txt \
            --model_path ${MODEL_DIR}/${version}/${topk}/
    done
done