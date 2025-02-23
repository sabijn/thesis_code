#!/bin/bash

set -e

MODEL_DIR=/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/pcfg-lm/resources/checkpoints/
MODEL=deberta
DATA_FILE=/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/corpora/eval.txt
OUTPUT_DIR=/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/mechanistic-interpret/results/bigram

python main.py --model_dir ${MODEL_DIR} \
                --model ${MODEL} \
                --top_k 1.0 \
                --data_file ${DATA_FILE} \
                --output_dir ${OUTPUT_DIR} \
                --log_probs False