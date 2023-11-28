import nltk

def read_trees_from_file(filename : str = 'corpora/eval_trees_10k.txt') -> list:
    """
    Read trees from txt file and transform into NLTK trees
    """
    trees = []
    sentences = []
    with open(filename) as f:
        lines = f.readlines()
        for i, example in enumerate(lines):
            # read tree from string
            tree = nltk.Tree.fromstring(example)
            sentence = tree.leaves()

            trees.append(tree)
            sentences.append(sentence)
    
    return trees, sentences

if __name__ == '__main__':
    """
    What's the plan:
    1. Create activations
        a. Remove punctuation?
        b. null elements?
    2. Create labels 
        a. Map 43 POS tags to te 7 base classes
        a. Find label of the LCA token pair
        b. The depth of the LCA (w_i, w_i+1) relative to the depth of (w_i-1, w_i)
        c. ... Geen idee
    """
    split = 10
    trees, sentences = read_trees_from_file(split=split)



