from transformers import AutoModelForMaskedLM
from pathlib import Path
from argparser import create_config_dict
from data import ExperimentManager
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
from torchmetrics.functional import accuracy
import os
import json
import logging
import sys
import random

logging.basicConfig(stream=sys.stdout,
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

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

if __name__ == "__main__":
    """
    Run script: python main.py --model.model_type deberta --data.data_dir corpora
    """
    config_dict = create_config_dict()
    pprint(config_dict)

    home_dir = Path(os.environ['CURRENT_WDIR'])
    if home_dir.exists():
        logger.debug("Home directory exists!")
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
            logger.info('Running on GPU.')
        # elif torch.backends.mps.is_available():
        #     # For running on M1
        #     device = torch.device("mps")
        #     logger.info('Running on M1 GPU.')
        else:
            # For running on laptop
            device = torch.device("cpu")
            logger.info('Running on CPU.')
    else:
        device = torch.device(config_dict['trainer']['device'])

    # Load model
    logger.info('Loading model...')
    OGmodel, tokenizer = load_model(model_path, device)
    OGmodel.eval()

    # Initiate experiments
    CurrentExperiment = ExperimentManager(config_dict)

    val_final = []
    test_final = []

    for layer_idx, states in tqdm(CurrentExperiment.activations.items()):
        logging.info(f"Training layer {layer_idx}...")
        save_name = f"{CurrentExperiment.base_name}_{layer_idx}"

        # with torch.no_grad():
        #     states = OGmodel.cls.predictions.transform(states.to(device))

        logging.info("Splitting data in train, dev and test sets.")
        X_train, y_train, X_dev, y_dev, X_test, y_test = CurrentExperiment.create_train_dev_test_split(layer_idx)
            
        ninp, nout = X_train.shape[-1], len(CurrentExperiment.label_vocab)

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

        logging.info("Started training...")
        trainer = pl.Trainer(default_root_dir=os.path.join(config_dict['experiments']['checkpoint_path'], save_name),                          
                                accelerator='mps' if device == 'mps' else 'cpu', 
                                devices=1,                                            
                                max_epochs=config_dict['trainer']['epochs'],                                                                    
                                callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")])

        pl.seed_everything(42)
        model = DiagModule(model_hparams={"num_inp":X_train.shape[-1], "num_units":len(CurrentExperiment.label_vocab)}, optimizer_hparams={"lr": config_dict['trainer']['lr']})
        trainer.fit(model, train_loader, dev_loader)

        model = DiagModule.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training

        # Test best model on validation and test set
        val_result = trainer.test(model, dev_loader, verbose=False)
        test_result = trainer.test(model, test_loader, verbose=False)
        
        val_final.append(val_result)
        test_final.append(test_result)

        result = {"test": test_result, "val": val_result}
        CurrentExperiment.results_file.write(f'Layer {layer_idx} \n {result}\n')

    CurrentExperiment.results_file.close()

    # write val_final and test_final to seperate pickle files
    with open(CurrentExperiment.val_results_file, 'wb') as f:
        pickle.dump(val_final, f)
    with open(CurrentExperiment.test_results_file, 'wb') as f:
        pickle.dump(test_final, f)
    print("Label vocab", CurrentExperiment.label_vocab)