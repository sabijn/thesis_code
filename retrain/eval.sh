#!/bin/bash

set -e

topks=("0.2" "0.3" "0.4" "0.5" "0.6")
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
                        --max_parse_time 10
    done
done
