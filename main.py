from transformers import AutoModelForMaskedLM
from pathlib import Path
from argparser import create_config_dict
from tokenizer import *
from pprint import pprint
import torch
import pickle
from tqdm import tqdm
import numpy as np

import pytorch_lightning as pl
import torch
from torch import nn
import torch.optim as optim
from pytorch_lightning.callbacks import ModelCheckpoint

import torchmetrics
# from torchmetrics.wrappers import ClasswiseWrapper
# from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.functional import accuracy
import os
import json
# from pytorch_lightning import Callback

class MyModule(nn.Module):
    def __init__(self, num_inp=768, num_units=18):
        super(MyModule, self).__init__()

        self.dense0 = nn.Linear(num_inp, num_units)

    def forward(self, X, **kwargs):
        return self.dense0(X)
                        
# class DiagModule(pl.LightningModule):
#     def __init__(self, model_hparams, optimizer_hparams):
#         """
#         Inputs:
#             model_hparams - Hyperparameters for the model, as dictionary.
#             optimizer_name - Name of the optimizer to use. Currently supported: Adam, SGD
#             optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
#         """
#         super().__init__()
#         # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
#         self.save_hyperparameters()
#         # Create model
#         self.model = MyModule(model_hparams['num_inp'], model_hparams['num_units'])
#         # Create loss module
#         self.loss_module = nn.CrossEntropyLoss()

#     def forward(self, x):
#         # Forward function that is run when visualizing the graph
#         return self.model(x)

#     def configure_optimizers(self):
#         optimizer = optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
#         return [optimizer]

#     def training_step(self, batch, batch_idx):
#         # "batch" is the output of the training data loader.
#         x, targets = batch
#         preds = self.model(x)
#         loss = self.loss_module(preds, targets)
#         acc = (preds.argmax(dim=-1) == targets).float().mean()

#         # Logs the accuracy per epoch to tensorboard (weighted average over batches)
#         self.log('train_acc', acc, on_step=False, on_epoch=True)
#         self.log('train_loss', loss)

#         return loss  # Return tensor to call ".backward" on

#     def validation_step(self, batch, batch_idx):
#         x, targets = batch
#         preds = self.model(x).argmax(dim=-1)
#         acc = (targets == preds).float().mean()
#         # By default logs it per epoch (weighted average over batches)

#         self.log('val_acc', acc)
#         # log classwise accuracy
    
    # def test_step(self, batch, batch_idx):
    #     x, targets = batch
    #     preds = self.model(x).argmax(dim=-1)
    #     acc = (targets == preds).float().mean()
    #     # By default logs it per epoch (weighted average over batches), and returns it afterwards
    #     self.log('test_acc', acc)

class DiagModule(pl.LightningModule):
    def __init__(self, model_hparams, optimizer_hparams):
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        # Create model
        self.model = MyModule(model_hparams['num_inp'], model_hparams['num_units'])
        # Create loss module
        self.loss_module = nn.CrossEntropyLoss()
        # Initialize dictionary to store classwise accuracy
        self.classwise_acc = {}
    
    def forward(self, x):
        # Forward function that is run when visualizing the graph
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
        return [optimizer]

    def training_step(self, batch, batch_idx):
        x, targets = batch
        preds = self.model(x)
        loss = self.loss_module(preds, targets)

        acc = (preds.argmax(dim=-1) == targets).float().mean()

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, targets = batch
        preds = self.model(x).argmax(dim=-1)

        acc = (targets == preds).float().mean()
        self.log('val_acc', acc)
        # Calculate classwise accuracy
        class_acc = torchmetrics.functional.accuracy(preds, targets, task='multiclass', num_classes=self.hparams.model_hparams['num_units'], average=None)
        # Convert tensor to numpy array
        class_acc = class_acc.cpu().numpy()

        # Update classwise accuracy in the log dictionary
        for i, acc in enumerate(class_acc):
            self.classwise_acc[f'class_{i}'] = acc

    def test_step(self, batch, batch_idx):
        x, targets = batch
        preds = self.model(x).argmax(dim=-1)

        acc = (targets == preds).float().mean()
        self.log('test_acc', acc)

        # Calculate classwise accuracy
        class_acc = torchmetrics.functional.accuracy(preds, targets, task='multiclass', num_classes=self.hparams.model_hparams['num_units'], average=None)
        # Convert tensor to numpy array
        class_acc = class_acc.cpu().numpy()

        # Update classwise accuracy in the log dictionary
        for i, acc in enumerate(class_acc):
            self.classwise_acc[f'class_{i}'] = acc

    def on_validation_epoch_end(self):
        # Log classwise accuracy at the end of each epoch
        self.log_dict(self.classwise_acc, on_epoch=True, prog_bar=True)
        self.classwise_acc = {}

    def on_test_epoch_end(self):
        # Log classwise accuracy at the end of testing
        self.log_dict(self.classwise_acc, on_epoch=True, prog_bar=True)
        self.classwise_acc = {}

def load_model(checkpoint, device):
    model = AutoModelForMaskedLM.from_pretrained(checkpoint).to(device)

    with open(f'{checkpoint}/added_tokens.json') as f:
        vocab = json.load(f)
    tokenizer = create_tf_tokenizer_from_vocab(vocab)

    return model, tokenizer
    
def create_labels(file, device, skip_unk_tokens=False):
    labels = []
    # open file with bies labels
    with open(file, 'r') as f:
        for line in f:
            labels.extend(line.strip().split())
    
    vocab = {
        l: idx
        for idx, l in enumerate(set(labels))
    }
    
    tokenized_labels = [
        vocab[l]
        for l in labels
    ]
    
    tokenized_labels = torch.tensor(tokenized_labels).to(device)

    return tokenized_labels, vocab

def create_control_task_labels(labels, device, vocab):
    """
    Randomly assign BIES labels to tokens based on distribution of labels in the training set.
    """
    # randomly assign bies labels based on distribution of labels in the training set
    p = []

    for uv in set(labels.tolist()):
        p.append(labels.tolist().count(uv))

    control_labels = np.random.choice(list(set(labels.tolist())), len(labels), p=[elem/sum(p) for elem in p])
    tokenized_control_labels = torch.tensor(control_labels).to(device)

    return tokenized_control_labels

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

    if config_dict['trainer']['device'] is None:
        if torch.cuda.is_available():
            # For running on snellius
            device = torch.device("cuda")
            print('Running on GPU.')
        # elif torch.backends.mps.is_available():
        #     # For running on M1
        #     device = torch.device("mps")
        #     print('Running on M1 GPU.')
        else:
            # For running on laptop
            device = torch.device("cpu")
            print('Running on CPU.')
    else:
        device = torch.device(config_dict['trainer']['device'])

    # Load model
    print('Loading model...')
    OGmodel, tokenizer = load_model(model_path, device)
    OGmodel.eval()

    # Load labels
    match config_dict['experiments']['type']:
        case 'chunking':
            print('Loading chuncking labels')
            label_path = 'data/train_bies_labels.txt'
            labels, label_vocab = create_labels(label_path, device, skip_unk_tokens=True)
 
            if config_dict['experiments']['control_task']:
                labels = create_control_task_labels(labels, device, label_vocab)
                results_file = open('results_chuncking_control.txt', 'w')
                base_name = 'chuncking/best_chuncking_control_layer'
            else:
                results_file = open('results_bies_per_class.txt', 'w')
                base_name = 'chuncking/best_bies_per_class_chuncking_layer'
            
            # check if activations are already generated, if not, generate them
            print('Loading activations for chuncking...')
            if Path('data/activations.pickle').exists():
                with open('data/activations.pickle', 'rb') as f:
                    activations = pickle.load(f)
            else:
                raise ValueError("Activations not found, please run create_activations.py first.")

        case 'lca':
            # contains labels for LCA task
            print('Loading LCA labels')
            label_path = 'data/train_rel_labels.txt'
            labels, label_vocab = create_labels(label_path, device, skip_unk_tokens=True)

            if config_dict['experiments']['control_task']:
                labels = create_control_task_labels(labels, device, label_vocab)
                results_file = open(f"results/results_lca_{config_dict['activations']['mode']}_control.txt", 'w')
                base_name = f"lca/best_lca_{config_dict['activations']['mode']}_control_layer"
            else:
                results_file = open(f"results/results_lca_{config_dict['activations']['mode']}.txt", 'w')
                base_name = f"lca/best_lca_{config_dict['activations']['mode']}_layer"

            # load activations
            print('Loading activations for LCA...')
            if Path(f"data/activations_combined_{config_dict['activations']['mode']}.pickle").exists():
                with open(f"data/activations_combined_{config_dict['activations']['mode']}.pickle", 'rb') as f:
                    activations = pickle.load(f)

                    for layer_idx, layer_states in activations.items():
                        activations[layer_idx] = torch.concat(layer_states)
            else:
                raise ValueError("Activations not found, please check your spelling or run create_activations.py first.")

            assert len(labels) == len(activations[0]), \
                f"Length of labels ({len(labels)}) does not match length of activations ({len(activations[0])})"

        case 'lca_tree':
            print('Loading LCA labels for tree task')
            label_path = 'data/train_rel_labels.txt'
            labels, label_vocab = create_labels(label_path, device, skip_unk_tokens=True)

            if config_dict['experiments']['control_task']:
                labels = create_control_task_labels(labels, device, label_vocab)
                results_file = open(f"results/results_lca_tree_control.txt", 'w')
                base_name = f"lca/best_lca_tree_control"
            else:
                results_file = open(f"results/results_lca_tree.txt", 'w')
                base_name = f"lca/best_lca_tree"
            
            # load activations
            print('Loading activations for LCA...')
            if Path(f"data/activations_concat_layers.pickle").exists():
                with open(f"data/activations_concat_layers.pickle", 'rb') as f:
                    activations = pickle.load(f)

                for layer_idx, layer_states in activations.items():
                    activations[layer_idx] = torch.concat(layer_states)
            else:
                raise ValueError("Activations not found, please check your spelling or run create_activations.py first.")

            assert len(labels) == len(activations[0]), \
                f"Length of labels ({len(labels)}) does not match length of activations ({len(activations[0])})"
        
        case 'shared_levels':
            print('Loading shared levels labels for tree task')
            label_path = 'data/train_shared_levels.txt'
            labels, label_vocab = create_labels(label_path, device, skip_unk_tokens=True)

            if config_dict['experiments']['control_task']:
                labels = create_control_task_labels(labels, device, label_vocab)
                results_file = open(f"results/results_shared_levels_control.txt", 'w')
                base_name = f"lca/best_shared_levels_control"
            else:
                results_file = open(f"results/results_shared_levels.txt", 'w')
                base_name = f"lca/best_shared_levels"
            
            # load activations
            print('Loading activations for shared levels...')
            if Path(f"data/activations_concat_layers.pickle").exists():
                with open(f"data/activations_concat_layers.pickle", 'rb') as f:
                    activations = pickle.load(f)

                for layer_idx, layer_states in activations.items():
                    activations[layer_idx] = torch.concat(layer_states)
            else:
                raise ValueError("Activations not found, please check your spelling or run create_activations.py first.")

            assert len(labels) == len(activations[0]), \
                f"Length of labels ({len(labels)}) does not match length of activations ({len(activations[0])})"

        case 'unary':
            print('Loading shared levels labels for tree task')
            label_path = 'data/train_unaries.txt'
            labels, label_vocab = create_labels(label_path, device, skip_unk_tokens=True)

            if config_dict['experiments']['control_task']:
                labels = create_control_task_labels(labels, device, label_vocab)
                results_file = open(f"results/results_unaries_control.txt", 'w')
                base_name = f"lca/best_unaries_control"
            else:
                results_file = open(f"results/results_unaries.txt", 'w')
                base_name = f"lca/best_unaries"
            
            # load activations
            print('Loading activations for unaries...')
            if Path(f"data/activations_concat_layers.pickle").exists():
                with open(f"data/activations_concat_layers.pickle", 'rb') as f:
                    activations = pickle.load(f)

                for layer_idx, layer_states in activations.items():
                    activations[layer_idx] = torch.concat(layer_states)
            else:
                raise ValueError("Activations not found, please check your spelling or run create_activations.py first.")

            assert len(labels) == len(activations[0]), \
                f"Length of labels ({len(labels)}) does not match length of activations ({len(activations[0])})"

    val_final = []
    test_final = []
    for layer_idx, states in tqdm(activations.items()):
        print(f"Training layer {layer_idx}...")
        save_name = f"{base_name}_{layer_idx}"

        with torch.no_grad():
            states = OGmodel.cls.predictions.transform(states.to(device))

        print("Loading data...")
        X_train, y_train, X_dev, y_dev, X_test, y_test = create_train_dev_test_split(states, labels)

        assert len(labels) == states.shape[0], \
                f"Length of labels ({len(labels)}) does not match length of activations ({states.shape[0]})"
            
        ninp, nout = X_train.shape[-1], len(label_vocab)

        trainset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                    shuffle=True,
                                    persistent_workers=True,
                                    batch_size=config_dict['trainer']['batch_size'],
                                    num_workers=config_dict['trainer']['num_workers'])
                                    # multiprocessing_context='fork' if torch.backends.mps.is_available() else None)
        
        devset = torch.utils.data.TensorDataset(X_dev, y_dev)
        dev_loader = torch.utils.data.DataLoader(dataset= devset, 
                                    shuffle=False,
                                    persistent_workers=True,
                                    batch_size=config_dict['trainer']['batch_size'],
                                    num_workers=config_dict['trainer']['num_workers'])
                                    # multiprocessing_context='fork' if torch.backends.mps.is_available() else None)
        
        testset = torch.utils.data.TensorDataset(X_test, y_test)
        test_loader = torch.utils.data.DataLoader(dataset= testset, 
                                    shuffle=False,
                                    persistent_workers=True,
                                    batch_size=config_dict['trainer']['batch_size'],
                                    num_workers=config_dict['trainer']['num_workers'])
                                    # multiprocessing_context='fork' if torch.backends.mps.is_available() else None)

        print("Started training...")
        trainer = pl.Trainer(default_root_dir=os.path.join(config_dict['experiments']['checkpoint_path'], save_name),                          
                                accelerator='mps' if device == 'mps' else 'cpu', 
                                devices=1,                                            
                                max_epochs=config_dict['trainer']['epochs'],                                                                    
                                callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")])

        pl.seed_everything(42)
        model = DiagModule(model_hparams={"num_inp":X_train.shape[-1], "num_units":len(label_vocab)}, optimizer_hparams={"lr": config_dict['trainer']['lr']})
        trainer.fit(model, train_loader, dev_loader)

        model = DiagModule.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training

        # Test best model on validation and test set
        val_result = trainer.test(model, dev_loader, verbose=False)
        test_result = trainer.test(model, test_loader, verbose=False)
        
        val_final.append(val_result)
        test_final.append(test_result)

        result = {"test": test_result, "val": val_result}
        results_file.write(f'Layer {layer_idx} \n {result}\n')

    # write val_final and test_final to seperate pickle files
    with open('val_final.pickle', 'wb') as f:
        pickle.dump(val_final, f)
    with open('test_final.pickle', 'wb') as f:
        pickle.dump(test_final, f)
    print(label_vocab)