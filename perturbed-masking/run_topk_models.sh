#!/bin/bash

# Exit immediately if any command exits with a non-zero status.
set -e

versions=("normal")
topks=("0.5" "0.6")
for topk in "${topks[@]}"; do
    for version in "${versions[@]}"; do
        echo "Running topk: $topk, version: $version"
        python main.py --metric dist \
                        --remove_punct \
                        --device cpu \
                        --model deberta \
                        --top_k "$topk" \
                        --embedding_layer \
                        --version "$version" \
                        --data /Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/reduce_grammar/corpora/ \
                        --home_model_path /Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/retrain/checkpoints/ \
                        --output_dir /Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/perturbed-masking/test_results/i_matrices/ \
                        --tree_path /Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/perturbed-masking/test_results/ \
                        --eval_results_dir /Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/perturbed-masking/test_results/eval/ \
                        --evaluation classic \
                        --all_layers \
                        --split 1000
    done
done


