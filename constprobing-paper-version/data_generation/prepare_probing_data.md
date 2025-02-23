# Prepare data and labels for probing

Here, labels and activations are created from different datasets.
Run prepare_probing_data.sh with the following parameters

str DATA_DIR: directory where the datasets are stored. The following structure is assumed: `datadir > treebank_size > trees.txt.example`
str TREEBANK_SIZE: size of the treebank used to generate the data
str HUB_MODEL_ID: huggingface id of the trained model. The treebank on which the model is trained has to correlate with the data used in these experiments. The possible models are listed under this.
str OUTPUT_DIR: directory of the output data (activations and labels). Only give the top directory.

## Models
- jumelet/gpt2_1t_1M_256d_8l
- jumelet/gpt2_5t_1M_256d_8l
- jumelet/gpt2_10t_1M_256d_8l
- jumelet/gpt2_50t_1M_256d_8l
- jumelet/gpt2_100t_1M_256d_8l
- jumelet/gpt2_500t_1M_256d_8l
- jumelet/gpt2_1000t_1M_256d_8l
- jumelet/gpt2_10000t_1M_256d_8l