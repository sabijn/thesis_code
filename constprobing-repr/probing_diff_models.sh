#!/bin/bash

# Exit immediately if any command exits with a non-zero status.
set -e

# Set the path to the edge probing repository.
MODEL=babyberta
PROBE=nonlinear
PROBING_DIR=/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/constprobing-repr
DATA_DIR=/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/constprobing-repr/data
MODEL_DIR=/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/retrain/checkpoints/${MODEL}
OUTPUT_DIR=/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/constprobing-repr/results

topks=("0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9")
versions=("normal")
experiments=("lca_tree" "shared_levels" "unary")
for version in "${versions[@]}"; do
    for topk in "${topks[@]}"; do
        for experiment in "${experiments[@]}"; do
            echo "Running probing experiments for topk=${topk} and version=${version}"
            python main.py --experiments.version ${version} \
                            --experiments.top_k ${topk} \
                            --model.model_type ${MODEL} \
                            --model.model_file ${MODEL_DIR}/${version}/${topk}/ \
                            --trainer.device cpu \
                            --data.rel_toks ${DATA_DIR}/${MODEL}/${version}/${topk}/train_rel_toks.txt \
                            --data.data_dir ${DATA_DIR}/${MODEL}/${version}/${topk} \
                            --data.output_dir ${OUTPUT_DIR}/${MODEL}/${version}/${PROBE}/${topk} \
                            --experiments.checkpoint_path ${PROBING_DIR}/models/${MODEL}/${version}/${PROBE}/${topk} \
                            --activations.output_dir ${DATA_DIR}/${MODEL}/${version}/${topk}/ \
                            --experiments.type ${experiment} \
                            --results.confusion_matrix \
                            --model.probe_type ${PROBE}
        done
    done
done