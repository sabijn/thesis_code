#!/bin/bash

# Exit immediately if any command exits with a non-zero status.
set -e

# Set the path to the edge probing repository.
MODEL=babyberta
DATA_DIR=/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/constprobing-repr/results
POS_DIR=/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/constprobing-repr/data

topks=("0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9")
versions=("normal")
for version in "${versions[@]}"; do
    for topk in "${topks[@]}"; do
        echo "Running probing experiments for topk=${topk} and version=${version}"
        current_output_dir=$DATA_DIR/$MODEL/$version/$topk/full_tree
        mkdir -p $current_outputdir
        python predictions_to_trees.py --lca $DATA_DIR/$MODEL/$version/$topk/lca_tree/predcitions_lca_tree.txt \
                                        --levels $DATA_DIR/$MODEL/$version/$topk/lca_tree/predcitions_shared_levels.txt \
                                        --out $current_output_dir/concat_test_trees.txt \
                                        --pos_text $POS_DIR/$MODEL/$version/$topk/train_POS_labels.txt \
                                        --unary $DATA_DIR/$MODEL/$version/$topk/unary/predcitions_unary.txt
    done
done