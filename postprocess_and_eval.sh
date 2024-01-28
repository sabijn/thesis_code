#!/bin/bash

model=$1
datadir="data/"
resultsdir="results/"

echo "STEP: model "$model

echo "STEP: Reconstruct trees from lca, level, unary predictions-"
python3 reconstruct_tree/predictions_to_trees.py -pos_text $datadir/test_POS.txt \
                    -lca $resultsdir/lca_tree/predictions_lca_tree.txt \
                    -levels $resultsdir/shared_levels/predictions_shared_levels.txt \
                    -unary $resultsdir/unary/predictions_unary.txt \
                    -out $resultsdir/full_tree/concat_test_trees.txt

# echo "STEP: evaluate trees"

# labeledoutput=$resultsdir"full_tree/concat_evalb_labeled.log"
# unlabeledoutput=$resultsdir"full_tree/concat_evalb_unlabeled.log"

# tree2labels/EVALB/evalb -p tree2labels/EVALb/COLLINS.prm $modeldir/concat_test_trees.txt $datadir/test_gold_trees.txt > $labeledoutput
# tree2labels/EVALB/evalb -p tree2labels/EVALb/COLLINS_unlabeled.prm $modeldir/concat_test_trees.txt $datadir/test_gold_trees.txt > $unlabeledoutput

# echo "STEP: Done with eval. Results are in the files: "
# echo $labeledoutput
# echo $unlabeledoutput