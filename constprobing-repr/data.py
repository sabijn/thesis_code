import torch
import numpy as np
import pickle
from pathlib import Path
import logging
from collections import Counter
import json

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
        self.constituents = config_dict['experiments']['constituents']
        self.rel_toks = self._read_rel_toks()
        self.sentences = self._read_sentences()
        self.label_path = self._set_label_path()
        self.labels, self.label_vocab, self.indices = self._create_labels()
        self._set_results_file()
        self.activations = self._load_activations()

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

    def _create_idx2class(self, vocab):
        return {v: k for k, v in vocab.items()}

    def _create_labels(self):
        logging.info(f'Creating labels for {self.name}')
        labels = []

        with open(self.label_path, 'r') as f:
            for line in f:
                labels.extend(line.strip().split())
        
        if self.name == 'ii':
            vocab = {l: idx for idx, l in enumerate(set(self.constituents))}
            tokenized_labels = torch.tensor([vocab[l] if l in list(vocab.keys()) else -1 for l in labels]).to(self.device)

            indices = [idx for idx, label in enumerate(labels) if label in self.constituents]
            tokenized_labels = torch.index_select(tokenized_labels, 0, torch.LongTensor(indices))
            
            return tokenized_labels, vocab, indices

        vocab = {l: idx for idx, l in enumerate(set(labels))}
        self.idx2class = self._create_idx2class(vocab)
        tokenized_labels = torch.tensor([vocab[l] for l in labels]).to(self.device)

        if self.config_dict['experiments']['control_task']:
            logging.info(f'Creating control task labels for {self.name}')
            tokenized_labels = self._create_control_task_labels(tokenized_labels, vocab)
        
        if self.config_dict['data']['sampling']:
            logging.info(f'Sampling data for {self.name}')
            tokenized_labels, indices = self._sample_data(tokenized_labels)

        elif self.config_dict['experiments']['type'] == 'ii':
            logging.info(f'Sampling balanced data for interventions')
            tokenized_labels, indices = self._sample_intervention_data(tokenized_labels, vocab)
            logging.info(f'Sampled {len(indices)} indices for interventions. Distribution: {tokenized_labels.unique(return_counts=True)}')
            
        else:
            indices = None

        return tokenized_labels, vocab, indices

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

        if self.name in ['lca_tree', 'shared_levels', 'unary', 'ii']:
            self.rel_toks_test = [self.rel_toks[idx] for idx in test_ids]

        if self.config_dict['data']['generate_test_data']:
            sent_idx, word_idx, _ = map(int, self.rel_toks_test[0].split('_'))

            with open(self.config_dict['data']['data_dir'] / f'test_start_idx.json', 'w') as f:
                json.dump({
                    'sent_idx': sent_idx,
                    'word_idx': word_idx
                }, f)

        return X_train, y_train, X_dev, y_dev, X_test, y_test
    

    def _load_activations(self):
        """
        1. Activations per layer
        2. Activations concatenated into one
        3. Sampled activations
        """
        failed = False

        logging.info(f'Loading activations for {self.name}')
        if self.name == 'chunking':
            activations_path = self.config_dict['activations']['output_dir'] / 'activations_concat_layers.pickle'
        
        elif self.name == 'lca' or self.name == 'ii':
            activations_path = self.config_dict['activations']['output_dir'] / 'activations_combined.pickle'
            
        elif self.name in ['lca_tree', 'shared_levels', 'unary']:
            activations_path = self.config_dict['activations']['output_dir'] / 'activations_layers_combined.pickle'
            
        else:
            logging.critical(f"This experiment is not supported yet: {self.name}.")
            failed = True

        if activations_path.exists():
            with open(activations_path, 'rb') as f:
                activations = pickle.load(f)
        else:
            logging.critical(f"Loading of activations failed, check path or generate with create_activations.py or combine_activations.py")
            failed = True
        
        if failed:
            raise ValueError
        

        if self.name in ['lca', 'lca_tree', 'shared_levels', 'unary'] and not self.config_dict['data']['sampling']:
            for layer_idx, layer_states in activations.items():
                activations[layer_idx] = torch.concat(layer_states)
        
        elif self.name in ['lca', 'lca_tree', 'shared_levels', 'unary'] and self.config_dict['data']['sampling']:
            for layer_idx, layer_states in activations.items():
                activations[layer_idx] = torch.index_select(torch.concat(layer_states), 0, torch.LongTensor(self.indices))
        
        elif self.name == 'ii':
            # interchange interventions probe is only trained on the last layer activations
            # such that interventions can be conducted on everything in between
            activations = {0: torch.index_select(torch.concat(activations[max(activations.keys())]), 0, torch.LongTensor(self.indices))}
        
        assert len(self.labels) == len(activations[0]), \
        f"Length of labels ({len(self.labels)}) does not match length of activations ({len(activations[0])})"

        return activations

    def _read_rel_toks(self):
        with open(self.config_dict['data']['rel_toks'], 'r') as f:
            rel_toks = f.readlines()
        
        return [tok.strip('\n') for sent in rel_toks for tok in sent.split(' ')]

    
    def _read_sentences(self):
        with open(self.config_dict['data']['data_dir'] / 'train_text.txt', 'r') as f:
            sentences = f.readlines()
        
        return [sent.strip('\n') for sent in sentences]
    

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
    

    def _set_label_path(self):
        if self.name == 'chunking':
            logging.info('Running chunking experiments')
            label_path = self.config_dict['data']['data_dir'] / 'train_bies_labels.txt'
        elif self.name == 'lca' or self.name == 'ii':
            logging.info('Running lca experiments')
            label_path = self.config_dict['data']['data_dir'] / 'train_rel_labels.txt'
        elif self.name == 'lca_tree':
            logging.info('Running lca for full reconstructing (this means, concatenated layers!)')
            label_path = self.config_dict['data']['data_dir'] / 'train_rel_labels.txt'
        elif self.name == 'shared_levels':
            if self.config_dict['data']['sampling']:
                logging.info('Running shared levels for full reconstructing with sampling (this means, concatenated layers!)')
                label_path = self.config_dict['data']['data_dir'] / 'train_shared_balanced.txt'
            else:
                logging.info('Running shared levels for full reconstructing WITHOUT sampling (this means, concatenated layers!)')
                label_path = self.config_dict['data']['data_dir'] / 'train_shared_levels.txt'
        elif self.name == 'unary':
            logging.info('Running unary experiments for full reconstruction.')
            label_path = self.config_dict['data']['data_dir'] / 'train_unaries.txt'
        else:
            logging.critical("This experiment is not supported yet.")
            raise ValueError('This experiment is not supported yet.')
        
        return label_path
    

    def _set_results_file(self):
        if self.config_dict['experiments']['control_task']:
            self.results_file = open(self.config_dict['data']['output_dir'] /f'{self.name}/results_control{self.config_dict["experiments"]["control_task"]}.txt', 'w')
            self.test_results_file = self.config_dict['data']['output_dir'] /f'{self.name}/test_results_control{self.config_dict["experiments"]["control_task"]}.pickle'
            self.val_results_file = self.config_dict['data']['output_dir'] /f'{self.name}/val_results_control{self.config_dict["experiments"]["control_task"]}.pickle'
            self.base_name = f'{self.name}/best_model_control_{self.name}'
        else:
            results_dir = self.config_dict['data']['output_dir'] / f'{self.name}'
            results_dir.mkdir(parents=True, exist_ok=True)
            self.results_file = open(results_dir / f'results_default_{self.config_dict["data"]["sampling_size"]}.txt', 'w')
            self.test_results_file = self.config_dict['data']['output_dir'] /f'{self.name}/test_results_default_{self.config_dict["data"]["sampling_size"]}.pickle'
            self.val_results_file = self.config_dict['data']['output_dir'] /f'{self.name}/val_results_default_{self.config_dict["data"]["sampling_size"]}.pickle'
            self.base_name = f'{self.name}/best_model_default_{self.name}'