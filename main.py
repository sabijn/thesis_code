from transformers import AutoModelForMaskedLM
from pathlib import Path
from argparser import create_config_dict
from tokenizer import load_pretrained_tokenizer
from extract_activations import extract_representations
from pprint import pprint
from data_probing import read_trees_from_file
import torch
import pickle
from pathlib import Path
from tqdm import tqdm


def load_transformer_models(path : Path, device : str = 'cpu'):
    tokenizer = load_pretrained_tokenizer(path / "added_tokens.json")
    model = AutoModelForMaskedLM.from_pretrained(path, output_hidden_states=True).to(device)

    return model, tokenizer

if __name__ == "__main__":
    """
    Run script: python main.py --model.model_type deberta --data.data_dir corpora
    """
    config_dict = create_config_dict()
    pprint(config_dict)

    home_dir = Path("/Users/sperdijk/Documents/Master/Jaar 3/Thesis/thesis_code/")
    if home_dir.exists():
        print("Home directory exists!")
    else:
        exit("Home directory does not exist!")

    if config_dict['model']['model_type'] == 'deberta':
        model_path = Path('pcfg-lm/resources/checkpoints/deberta/')
    elif config_dict['model']['model_type'] == 'gpt2':
        model_path = Path('pcfg-lm/resources/checkpoints/gpt2/')

    if torch.cuda.is_available():
        config_dict['model']['device'] = torch.device("cuda")
        print('Running on GPU.')
    else:
        config_dict['model']['device'] = torch.device("cpu")
        print('Running on CPU.')

    print("Loading models...")
    model, tokenizer = load_transformer_models(home_dir / model_path, config_dict['model']['device'])

    print("Loading data...")
    # datasets = load_data(tokenizer, **config_dict['data'])
    # print(datasets['train'][0])

    trees, sentences = read_trees_from_file()

    # Test example
    # model.eval()
    # with torch.no_grad():
    #     inp = tokenizer(sentences[0], return_tensors='pt')
    #     out = model(**inp, output_hidden_states=True)
    #     print(out.hidden_states[-1].shape)

    # check if file exists
    if Path('data/activations.pickle').exists():
        print("Loading activations...") 
        with open('data/activations.pickle', 'rb') as f:
            activations = pickle.load(f)
    else:
        activations = extract_representations(model, tokenizer, config_dict['data']['data_dir'] / config_dict['data']['eval_file'],
                                              config_dict['model']['device'], config_dict['activations']['dtype'])
        with open('data/activations.pickle', 'wb') as f: 
            pickle.dump(activations, f)
    
    print("Done!")