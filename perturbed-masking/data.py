import nltk

word_tags = ['COLON', 'TO', 'TICK', 'NN', 'JJR', 'PDT', 'CD', 'RBS', 'NNP', 'JJS', 'SYM', 'VBN', 'POS', 'FW', 'PRPDOLLAR', 
            'COMMA', 'WPDOLLAR', 'WDT', 'RB', 'WP', 'EX', 'RP', 'DT', 'RRB', 'JJ', 'APOSTROPHE', 'NNPS', 'NNS', 'RBR', 'PRP', 
            'VB', 'CC', 'VBZ', 'HASH', 'MD', 'IN', 'WRB', 'UH', 'LRB', 'VBD', 'VBP', 'VBG', 'DOT']

class Corpus():
    def __init__(self, dataset, split=None):
        self.split = split
        self.sens = []
        self.trees = []
        self.nltk_trees = []

        _ = self._load_data(dataset)
        
    def _load_data(self, dataset):
        with open(dataset, 'r') as f:
            for line in f:
                tree = nltk.Tree.fromstring(line)
                self.trees.append(self.tree2list(tree))
                self.nltk_trees.append(tree)
                self.sens.append(tree.leaves())

        if self.split:
            corpus_size = len(self.sens)
            _, dev_split, test_split = int(0.9 * corpus_size), corpus_size
            self.sens = self.sens[dev_split:test_split]
            self.trees = self.trees[dev_split:test_split]
            self.nltk_trees = self.nltk_trees[dev_split:test_split]
    
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
    
