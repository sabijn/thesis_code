"""
For creating combined activations (for lca and shared levels):
    - input file is activations.pickle (list with a tensor per sentence (words x embedding))
    - output file activations_combined.pickle

For concatenating these combined activations (for lca tree and shared levels)
    - input file is activations_combined.pickle (see above)
    - output file activations_layers_combined.pickle
"""

set -e

# Set the path to the edge probing repository.
DATA_DIR=/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code/constprobing-repr/data
MODEL=babyberta
topks=("0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9")
versions=("normal")
for version in "${versions[@]}"; do
    for topk in "${topks[@]}"; do
        echo "Combining activations for topk=${topk} and version=${version}"
        current_data_dir=$DATA_DIR/$MODEL/$version/$topk
        python combine_activations.py --input_file $current_data_dir/activations_combined.pickle \
                                        --output_file $current_data_dir/activations_layers_combined.pickle \
                                        --rel_toks $current_data_dir/train_rel_toks.txt \
                                        --mode concat \
                                        --concatenate_layers
    done
done