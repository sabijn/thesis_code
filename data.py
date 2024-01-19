import torch
import numpy as np
import pickle
from pathlib import Path
import logging
from collections import Counter

logger = logging.getLogger(__name__)

class ExperimentManager():
    def __init__(self, config_dict):
        """
        If you add a new experiment:
        1. Add the label path to the _set_label_path method
        2. Add the experiment in the right category in _load_activations
        """
        self.config_dict = config_dict
        self.name = config_dict['experiments']['type']
        self.device = config_dict['trainer']['device']
        self.label_path = self._set_label_path()
        self.labels, self.label_vocab, self.indices = self._create_labels()
        self._set_results_file()
        self.activations = self._load_activations()

    def _set_label_path(self):
        match self.name:
            case 'chunking':
                logging.info('Running chunking experiments')
                label_path = 'data/train_bies_labels.txt'
            case 'lca':
                logging.info('Running lca experiments')
                label_path = 'data/train_rel_labels.txt'
            case 'lca_tree':
                logging.info('Running lca for full reconstructing (this means, concatenated layers!)')
                label_path = 'data/train_rel_labels.txt'
            case 'shared_levels':
                if self.config_dict['data']['sampling']:
                    logging.info('Running shared levels for full reconstructing with sampling (this means, concatenated layers!)')
                    label_path = 'data/train_shared_balanced.txt'
                else:
                    logging.info('Running shared levels for full reconstructing WITHOUT sampling (this means, concatenated layers!)')
                    label_path = 'data/train_shared_levels.txt'
            case 'unary':
                logging.info('Running unary experiments for full reconstruction.')
                label_path = 'data/train_unaries.txt'
        
        return label_path

    def _create_labels(self):
        logging.info(f'Creating labels for {self.name}')
        labels = []

        with open(self.label_path, 'r') as f:
            for line in f:
                labels.extend(line.strip().split())
        
        vocab = {l: idx for idx, l in enumerate(set(labels))}
        
        tokenized_labels = torch.tensor([vocab[l] for l in labels]).to(self.device)

        if self.config_dict['experiments']['control_task']:
            logging.info(f'Creating control task labels for {self.name}')
            tokenized_labels = self._create_control_task_labels(tokenized_labels, vocab)
        
        if self.config_dict['data']['sampling']:
            logging.info(f'Sampling data for {self.name}')
            tokenized_labels, indices = self._sample_data(tokenized_labels)
        else:
            indices = None

        return tokenized_labels, vocab, indices

    def _create_control_task_labels(self):
        """
        Randomly assign BIES labels to tokens based on distribution of labels in the training set.
        """
        # randomly assign bies labels based on distribution of labels in the training set
        p = []

        for uv in set(self.labels.tolist()):
            p.append(self.labels.tolist().count(uv))

        control_labels = np.random.choice(list(set(self.labels.tolist())), len(self.labels), p=[elem/sum(p) for elem in p])
        tokenized_control_labels = torch.tensor(control_labels).to(self.device)

        return tokenized_control_labels

    def create_train_dev_test_split(self, idx, train_size=0.8, dev_size=0.9):
        states = self.activations[idx]
        total_size = len(states)

        train_idx, dev_idx, test_idx = int(total_size * train_size), int(total_size * dev_size), total_size
        
        train_ids = range(0, train_idx)
        dev_ids = range(train_idx, dev_idx)
        test_ids = range(dev_idx, test_idx)

        X_train = states[train_ids]
        y_train = self.labels[train_ids]

        X_dev = states[dev_ids]
        y_dev = self.labels[dev_ids]

        X_test = states[test_ids]
        y_test = self.labels[test_ids]

        return X_train, y_train, X_dev, y_dev, X_test, y_test

    def _set_results_file(self):
        if self.config_dict['experiments']['control_task']:
            self.results_file = open(f'results/{self.name}/results_control{self.config_dict["experiments"]["control_task"]}.txt', 'w')
            self.test_results_file = f'results/{self.name}/test_results_control{self.config_dict["experiments"]["control_task"]}.pickle'
            self.val_results_file = f'results/{self.name}/val_results_control{self.config_dict["experiments"]["control_task"]}.pickle'
            self.base_name = f'{self.name}/best_model_control_{self.name}'
        else:
            self.results_file = open(f'results/{self.name}/results_default_{self.config_dict["data"]["sampling_size"]}.txt', 'w')
            self.test_results_file = f'results/{self.name}/test_results_default_{self.config_dict["data"]["sampling_size"]}.pickle'
            self.val_results_file = f'results/{self.name}/val_results_default_{self.config_dict["data"]["sampling_size"]}.pickle'
            self.base_name = f'{self.name}/best_model_default_{self.name}'
    
    def _sample_data(self, labels):
        # sample data to obtain balanced classes
        class_idx = Counter(labels.tolist())

        all_indices = []
        for count in class_idx.values():
            if count > self.config_dict['data']['sampling_size']:
                indices = list(np.random.choice(count, self.config_dict['data']['sampling_size'], replace=False))
                all_indices.extend(indices)
        
        sampled_labels = torch.tensor([labels[idx] for idx in all_indices]).to(self.device)
    
        return sampled_labels, all_indices

    def _load_activations(self):
        """
        1. Activations per layer
        2. Activations concatenated into one
        3. Sampled activations
        """
        if self.name in ['chuncking', 'lca', 'ii']:
            logging.info(f'Loading activations for {self.name}')
            if Path('data/activations.pickle').exists():
                with open('data/activations.pickle', 'rb') as f:
                    activations = pickle.load(f)
            else:
                logging.critical("Activations not found, please run create_activations.py first.")
                raise ValueError("Activations not found, please run create_activations.py first.")
            
        elif self.name in ['lca_tree', 'shared_levels', 'unary']:
            logging.info(f'Loading activations for {self.name} without sampling')
            if Path(f"data/activations_concat_layers.pickle").exists():
                with open(f"data/activations_concat_layers.pickle", 'rb') as f:
                    activations = pickle.load(f)

                if self.config_dict['data']['sampling']:
                    for layer_idx, layer_states in activations.items():
                        activations[layer_idx] = torch.index_select(torch.concat(layer_states), 0, torch.LongTensor(self.indices))
                else:
                    for layer_idx, layer_states in activations.items():
                        activations[layer_idx] = torch.concat(layer_states)
            else:
                logging.critical("Activations not found, please run create_activations.py first.")
                raise ValueError("Activations not found, please check your spelling or run create_activations.py first.")
            
        else:
            logging.critical("This experiment is not supported yet.")
            raise ValueError('This experiment is not supported yet.')

        assert len(self.labels) == len(activations[0]), \
                f"Length of labels ({len(self.labels)}) does not match length of activations ({len(activations[0])})"

        return activations