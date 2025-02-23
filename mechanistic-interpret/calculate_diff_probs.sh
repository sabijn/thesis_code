#!/bin/bash

set -e

MODEL=deberta
DATA_FILE=/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/mechanistic-interpret/results
OUTPUT_DIR=/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/mechanistic-interpret/results/diffs
PCFG_DIR=/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/mechanistic-interpret/pcfg_probs/uniform

python calculate_diff_probs.py --model ${MODEL} \
                                --top_k 1.0 \
                                --data_dir ${DATA_FILE} \
                                --output_dir ${OUTPUT_DIR} \
                                --pcfg_dir ${PCFG_DIR} \
                                --log_probs True \
                                --ngram uniform
                