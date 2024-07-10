#!/bin/bash

# Exit immediately if any command exits with a non-zero status.
set -e

# Set the path to the edge probing repository.
MODEL=babyberta
PROBE=linear
PROBING_DIR=/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/constprobing-repr
DATA_DIR=/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/constprobing-repr/data
MODEL_DIR=/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/retrain/checkpoints_v2/${MODEL}
OUTPUT_DIR=/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/constprobing-repr/results_v2

topks=("0.2" "0.3" "0.4" "0.5")
versions=("normal")
experiments=("lca" "chunking")
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