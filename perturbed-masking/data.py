from typing import Any
import nltk

from classes import (TokenizerConfig, 
                        Tokenizer, 
                        PCFGConfig,
                        PCFG)

word_tags = ['COLON', 'TO', 'TICK', 'NN', 'JJR', 'PDT', 'CD', 'RBS', 'NNP', 'JJS', 'SYM', 'VBN', 'POS', 'FW', 'PRPDOLLAR', 
            'COMMA', 'WPDOLLAR', 'WDT', 'RB', 'WP', 'EX', 'RP', 'DT', 'RRB', 'JJ', 'APOSTROPHE', 'NNPS', 'NNS', 'RBR', 'PRP', 
            'VB', 'CC', 'VBZ', 'HASH', 'MD', 'IN', 'WRB', 'UH', 'LRB', 'VBD', 'VBP', 'VBG', 'DOT']

class Corpus():
    def __init__(self, corpus_file, grammar_file, split=1000):
        # self.sens = []
        # self.trees = []
        # self.nltk_trees = []

        self.dataset, _ = self._load_from_file(corpus_file, grammar_file)
        string_sens = self.dataset.test_corpus[:split]
        self.nltk_trees = [self.dataset.tree_corpus[s] for s in string_sens]
        self.trees = [self.tree2list(tree) for tree in self.nltk_trees]
        self.sens = [tree.leaves() for tree in self.nltk_trees]
        
    # def _load_data(self, dataset):
    #     with open(dataset, 'r') as f:
    #         for line in f:
    #             tree = nltk.Tree.fromstring(line)
    #             self.trees.append(self.tree2list(tree))
    #             self.nltk_trees.append(tree)
    #             self.sens.append(tree.leaves())

    #     if self.split and type(self.split) == float:
    #         corpus_size = len(self.sens)
    #         dev_split, test_split = int(0.9 * corpus_size), corpus_size
    #         self.sens = self.sens[dev_split:test_split]
    #         self.trees = self.trees[dev_split:test_split]
    #         self.nltk_trees = self.nltk_trees[dev_split:test_split]
        
    #     elif self.split and type(self.split) == int:
    #         self.sens = self.sens[:self.split]
    #         self.trees = self.trees[:self.split]
    #         self.nltk_trees = self.nltk_trees[:self.split]
    
    def _load_from_file(self, corpus_file, grammar_file, encoder="transformer", size=None):
        tokenizer_config = TokenizerConfig(
                add_cls=(encoder == "transformer"),
                masked_lm=(encoder == "transformer"),
                unk_threshold=5,
            )
        
        tokenizer = Tokenizer(tokenizer_config)

        config = PCFGConfig(
            is_binary=False,
            min_length=6,
            max_length=25,
            max_depth=25,
            corpus_size=size,
            grammar_file=grammar_file,
            start="S_0",
            masked_lm=(encoder == "transformer"),
            allow_duplicates=True,
            split_ratio=(0.8,0.1,0.1),
            use_unk_pos_tags=True,
            verbose=True,
            store_trees=True,
            output_dir='.',
            file=corpus_file
        )

        language = PCFG(config, tokenizer)

        return language, tokenizer

    def tree2list(self, tree):
        if isinstance(tree, nltk.Tree):
            if tree.label().split('_')[0] in word_tags:
                return tree.leaves()[0]
            else:
                root = []
                for child in tree:
                    c = self.tree2list(child)
                    if c != []:
                        root.append(c)
                if len(root) > 1:
                    return root
                elif len(root) == 1:
                    return root[0]
        return []