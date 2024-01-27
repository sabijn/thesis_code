#!/bin/bash

model=$1
datadir="results/"

echo "STEP: model "$model

echo "STEP: Reconstruct trees from lca, level, unary predictions -"
python3 predictions_to_trees.py -pos_text $datadir/test_pos_and_words.txt \
                    -lca $modeldir/concat_lca/pred_labels_test.txt \
                    -levels $modeldir/concat_lev/pred_labels_test.txt \
                    -unary $modeldir/concat_unary/pred_labels_test.txt \
                    -out $modeldir/concat_test_trees.txt

echo "STEP: evaluate trees"

labeledoutput=$modeldir"concat_evalb_labeled.log"
unlabeledoutput=$modeldir"concat_evalb_unlabeled.log"

tree2labels/EVALB/evalb -p tree2labels/EVALb/COLLINS.prm $modeldir/concat_test_trees.txt $datadir/test_gold_trees.txt > $labeledoutput
tree2labels/EVALB/evalb -p tree2labels/EVALb/COLLINS_unlabeled.prm $modeldir/concat_test_trees.txt $datadir/test_gold_trees.txt > $unlabeledoutput

echo "STEP: Done with eval. Results are in the files: "
echo $labeledoutput
echo $unlabeledoutput