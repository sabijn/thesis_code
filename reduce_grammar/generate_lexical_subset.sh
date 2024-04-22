#!/bin/bash

# Exit immediately if any command exits with a non-zero status.
set -e

topks=("0.2" "0.3" "0.4" "0.5 "0.6 "0.7" "0.8" "0.9")
for topk in "${topks[@]}"; do
    python generate_subset.py --pcfg_dir=/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/reduce_grammar/grammars/nltk/nltk_pcfg.txt \
                        --output_dir=/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/reduce_grammar/grammars/nltk/lexical \
                        --top_k="$topk" \
                        --save \
                        --lexical
done