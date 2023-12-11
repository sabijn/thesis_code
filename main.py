from transformers import AutoModelForMaskedLM
from pathlib import Path
from argparser import create_config_dict
from tokenizer import *
from pprint import pprint
from data_probing import read_trees_from_file
import torch
import pickle
from pathlib import Path
from tqdm import tqdm
import numpy as np

from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import json
import random
from collections import defaultdict

class MyModule(nn.Module):
    def __init__(self, num_inp=768, num_units=18):
        super(MyModule, self).__init__()

        self.dense0 = nn.Linear(num_inp, num_units)

    def forward(self, X, **kwargs):
        return self.dense0(X)
                        
class DiagModule(pl.LightningModule):
    def __init__(self, model_hparams, optimizer_hparams):
        """
        Inputs:
            model_hparams - Hyperparameters for the model, as dictionary.
            optimizer_name - Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
        """
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        # Create model
        self.model = MyModule(model_hparams['num_inp'], model_hparams['num_units'])
        # Create loss module
        self.loss_module = nn.CrossEntropyLoss()

    def forward(self, x):
        # Forward function that is run when visualizing the graph
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
        return [optimizer]

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        x, targets = batch
        preds = self.model(x)
        loss = self.loss_module(preds, targets)
        acc = (preds.argmax(dim=-1) == targets).float().mean()

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        self.log('train_loss', loss)
        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        x, targets = batch
        preds = self.model(x).argmax(dim=-1)
        acc = (targets == preds).float().mean()
        # By default logs it per epoch (weighted average over batches)
        self.log('val_acc', acc)

    def test_step(self, batch, batch_idx):
        x, targets = batch
        preds = self.model(x).argmax(dim=-1)
        acc = (targets == preds).float().mean()
        # By default logs it per epoch (weighted average over batches), and returns it afterwards
        self.log('test_acc', acc)


def load_model(checkpoint, device):
    model = AutoModelForMaskedLM.from_pretrained(checkpoint).to(device)

    with open(f'{checkpoint}/added_tokens.json') as f:
        vocab = json.load(f)
    tokenizer = create_tf_tokenizer_from_vocab(vocab)

    return model, tokenizer
    
def create_bies_labels(file, device, skip_unk_tokens=False):
    bies_labels = []
    # open file with bies labels
    with open(file, 'r') as f:
        for line in f:
            bies_labels.extend(line.strip().split())
    print(len(bies_labels))
    
    bies_vocab = {
        bies: idx
        for idx, bies in enumerate(set(bies_labels))
    }
    
    tokenized_bies_labels = [
        bies_vocab[bies]
        for bies in bies_labels
    ]
    
    tokenized_bies_labels = torch.tensor(tokenized_bies_labels).to(device)

    return tokenized_bies_labels, bies_vocab

def create_train_dev_test_split(activations, bies_labels, train_size=0.8, dev_size=0.9):
    total_size = len(activations)
    train_idx, dev_idx, test_idx = int(total_size * train_size), int(total_size * dev_size), total_size
    
    train_ids = range(0, train_idx)
    dev_ids = range(train_idx, dev_idx)
    test_ids = range(dev_idx, test_idx)

    X_train = activations[train_ids]
    y_train = bies_labels[train_ids]

    X_dev = activations[dev_ids]
    y_dev = bies_labels[dev_ids]

    X_test = activations[test_ids]
    y_test = bies_labels[test_ids]

    return X_train, y_train, X_dev, y_dev, X_test, y_test

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
        # For running on snellius
        device = torch.device("cuda")
        print('Running on GPU.')
    elif torch.backends.mps.is_available():
        # For running on M1
        device = torch.device("mps")
        print('Running on M1 GPU.')
    else:
        # For running on laptop
        device = torch.device("cpu")
        print('Running on CPU.')

    # trees, sentences = read_trees_from_file()

    # Load model
    OGmodel, tokenizer = load_model(model_path, device)
    OGmodel.eval()

    # check if activations are already generated, if not, generate them
    print('Loading activations...')
    if Path('data/activations.pickle').exists():
        with open('data/activations.pickle', 'rb') as f:
            activations = pickle.load(f)
    else:
        raise ValueError("Activations not found, please run create_activations.py first.")

    label_path = 'data/train_bies_labels.txt'
    bies_labels, bies_vocab = create_bies_labels(label_path, device, skip_unk_tokens=True)

    results_file = open('results.txt', 'w')

    for layer_idx, states in tqdm(activations.items()):
        print(f"Training layer {layer_idx}...")
        save_name = f"best_model_layer_{layer_idx}"

        with torch.no_grad():
            states = OGmodel.cls.predictions.transform(states.to(device))

        print("Loading data...")
        X_train, y_train, X_dev, y_dev, X_test, y_test = create_train_dev_test_split(states, bies_labels)

        assert len(bies_labels) == states.shape[0], \
                f"Length of labels ({len(bies_labels)}) does not match length of activations ({states.shape[0]})"
            
        ninp, nout = X_train.shape[-1], len(bies_vocab)

        trainset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                    shuffle=True,
                                    batch_size=128,
                                    num_workers=8,
                                    multiprocessing_context='fork' if torch.backends.mps.is_available() else None)
        
        devset = torch.utils.data.TensorDataset(X_dev, y_dev)
        dev_loader = torch.utils.data.DataLoader(dataset= devset, 
                                    shuffle=False,
                                    batch_size=128,
                                    num_workers=8,
                                    multiprocessing_context='fork' if torch.backends.mps.is_available() else None)
        
        testset = torch.utils.data.TensorDataset(X_test, y_test)
        test_loader = torch.utils.data.DataLoader(dataset= testset, 
                                    shuffle=False,
                                    batch_size=128,
                                    num_workers=8,
                                    multiprocessing_context='fork' if torch.backends.mps.is_available() else None)

        print("Started training...")
        trainer = pl.Trainer(default_root_dir=os.path.join(config_dict['experiments']['checkpoint_path'], save_name),                          
                                accelerator="mps", 
                                devices=1,                                            
                                max_epochs=10,                                                                    
                                callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")])

        pl.seed_everything(42)
        model = DiagModule(model_hparams={"num_inp":X_train.shape[-1], "num_units":len(bies_vocab)+1}, optimizer_hparams={"lr": 1e-3})
        trainer.fit(model, train_loader, dev_loader)
        model = DiagModule.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training

        # Test best model on validation and test set
        val_result = trainer.test(model, dev_loader, verbose=False)
        test_result = trainer.test(model, test_loader, verbose=False)
        result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}
        results_file.write(f'Layer {layer_idx} \n {result}\n')
        print(result)