#!/bin/bash

# Exit immediately if any command exits with a non-zero status.
set -e

# Set the path to the edge probing repository.
MODEL=babyberta
DATA_DIR=/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/constprobing-repr/results_v2
GOLD_DIR=/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/constprobing-repr/data
EVALB_DIR=/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/constprobing-repr

topks=("0.2" "0.3" "0.4" "0.5")
versions=("normal")
for version in "${versions[@]}"; do
    for topk in "${topks[@]}"; do
        echo "Creating trees for topk=${topk} and version=${version}"
        current_output_dir=$DATA_DIR/$MODEL/$version/linear/$topk
        current_data_dir=$GOLD_DIR/$MODEL/$version/$topk
        mkdir -p $current_output_dir

        echo "STEP: evaluate trees"

        labeledoutput=$current_output_dir/full_tree/concat_evalb_labeled.log
        unlabeledoutput=$current_output_dir/full_tree/concat_evalb_unlabeled.log

        $EVALB_DIR/EVALB/evalb -p $EVALB_DIR/EVALB/COLLINS.prm $current_output_dir/full_tree/concat_test_trees.txt $current_data_dir/gold_trees_cleaned.txt > $labeledoutput
        $EVALB_DIR/EVALB/evalb -p $EVALB_DIR/EVALB/COLLINS_unlabeled.prm $current_output_dir/full_tree/concat_test_trees.txt $current_data_dir/gold_trees_cleaned.txt > $unlabeledoutput

        echo "STEP: Done with eval. Results are in the files: "
        echo $labeledoutput
        echo $unlabeledoutput
    done
done