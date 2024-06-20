import argparse
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='babyberta', type=str, choices=['deberta', 'gpt2', 'babyberta'])
    parser.add_argument('--model_dir', type=Path, required=True)
    parser.add_argument('--probing_dir', type=Path, required=True) 

    args = parser.parse_args()

    # Load the main models (MODEL_DIR)
    # Load the probing model (PROBING_DIR)
    # Load the data
    # Wait wat are you going to feed to the model to create the activations
    # Save place of NN or VERB and swap these later on (make pairs in the dataset)
    # Only with your test set


    
