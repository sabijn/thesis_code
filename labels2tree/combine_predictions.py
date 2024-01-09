from pathlib import Path
import nltk 
import pickle

def get_postag_trees(tree):
    """
    Gets a list of the PoS tags from the tree
    @return A list containing (word, postag)
    """
    postags = []
    
    for _, child in enumerate(tree):
        if len(child) == 1 and type(child[-1]) == type(""):
            word = child.leaves()[0]
            label = child.label().split("_")[0]
            postags.append((word, label))
        else:
            postags.extend(get_postag_trees(child))
    
    return postags

if __name__ == '__main__':
    home_path = Path("/Users/sperdijk/Documents/Master/Jaar 3/Thesis/thesis_code")
    labels = []

    with open(home_path / Path('data/train_rel_labels.txt')) as f:
        # LCA labels
        lca_labels = [l.strip() for l in f]
        labels.append(lca_labels)
    
    with open(home_path / Path('data/train_unaries.txt')) as f:
        # unary labels
        unary_labels = [l.strip() for l in f]
        labels.append(unary_labels)
    
    with open(home_path / Path('data/train_shared_levels.txt')) as f:
        # shared levels
        shared_levels = [l.strip() for l in f]
        labels.append(shared_levels)
    
    with open(home_path / Path('data/train_text.txt')) as f:
        # sentences
        sentences = [l.strip() for l in f]
    
    for (l1, l2, l3) in zip(lca_labels, unary_labels, shared_levels):
        assert len(l1.split()) == len(l2.split()) == len(l3.split()), f'{len(l1.split())} {len(l2.split())} {len(l3.split())}'
    # assert len(lca_labels) == len(unary_labels) == len(shared_levels) == len(sentences)

    # input for function: [[(LEVEL_LABEL_[UNARY_CHAIN]), ...], [...]]
    output_preds = ""
    for (lca, u, rel) in zip(lca_labels, unary_labels, shared_levels):
        line = ""
        for i, (lca_, u_, rel_) in enumerate(zip(lca.split(), u.split(), rel.split())):
            if i == 0:
                line += f'{lca_}@{rel_}@{u_}'
            else:
                line += f' {lca_}@{rel_}@{u_}'
        line += '\n'
        output_preds += line

    # write output_preds to file
    with open(home_path / Path('data/combined_predictions.txt'), 'w') as f:
        f.write(output_preds)
    
    sentences = []
    with open(home_path / Path('corpora/eval_trees_10k.txt')) as f:
        tree_corpus = [nltk.Tree.fromstring(l.strip()) for l in f]
        for tree in tree_corpus:
            sentences.append(get_postag_trees(tree))

    # write sentences to pickle
    with open(home_path / Path('data/sentences_postags.pickle'), 'wb') as f:
        pickle.dump(sentences, f)
    
    
    
