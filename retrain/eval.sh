#!/bin/bash

set -e

topks=("0.2" "0.3" "0.4" "0.5")
versions=("normal")
for version in "${versions[@]}"; do
    for topk in "${topks[@]}"; do
        echo "Calculating optimal and model perplexity for babyberta"
        python eval.py --model babyberta \
                        --version $version \
                        --top_k $topk \
                        --device cpu \
                        --size 10_000 \
                        --optimal \
                        --max_parse_time 10 \
                        --output_dir /Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/retrain/results_v2 \
                        --output_file_pcfg /Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/reduce_grammar/perplexities/babyberta/optimal_ppl_v4 \
                        --model_dir checkpoints_v2
    done
done
