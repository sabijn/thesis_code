from ete3 import Tree as EteTree
import torch
from tqdm import tqdm
import nltk
import logging

logger = logging.getLogger(__name__)


class FancyTree(EteTree):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, format=1, **kwargs)
        
    def __str__(self):
        return self.get_ascii(show_internal=True)
    
    def __repr__(self):
        return str(self)


def rec_tokentree_to_ete(tokentree):
    idx = str(tokentree.token["id"])
    children = tokentree.children
    if children:
        return f"({','.join(rec_tokentree_to_ete(t) for t in children)}){idx}"
    else:
        return idx
    
def tokentree_to_ete(tokentree):
    newick_str = rec_tokentree_to_ete(tokentree)

    return FancyTree(f"{newick_str};")

def create_gold_distances(corpus):
    all_distances = []

    for item in tqdm(corpus):
        tokentree = item.to_tree()
        ete_tree = tokentree_to_ete(tokentree)
        # nltk_tree = tokentree_to_nltk(tokentree)
        sen_len = len(ete_tree.search_nodes())
        distances = torch.zeros((sen_len, sen_len))

        for node1 in ete_tree.search_nodes():
            for node2 in ete_tree.search_nodes():
                distances[int(node1.name)-1][int(node2.name)-1] = node1.get_distance(node2)

        all_distances.append(distances)

    return all_distances

def create_pred_distances(corpus):
    all_distances = []

    for ete_tree in tqdm(corpus):
        sen_len = len(ete_tree.search_nodes())
        distances = torch.zeros((sen_len, sen_len))

        for node1 in ete_tree.search_nodes():
            for node2 in ete_tree.search_nodes():
                distances[int(node1.name)-1][int(node2.name)-1] = node1.get_distance(node2)

        all_distances.append(distances)

    return all_distances

def _nestedlist2nestedtuple(nestedlist):
    return tuple(map(_nestedlist2nestedtuple, nestedlist)) if isinstance(nestedlist, list) else nestedlist

def tree2etetree(nestedtuple):
    tree_format = str(_nestedlist2nestedtuple(nestedtuple))
    if ';' in tree_format or ':' in tree_format:
        tree_format = tree_format.replace(';', ',')
        tree_format = tree_format.replace(':', ',')
    etetree = EteTree((tree_format + ';'))

    return etetree

def nltk_tree_to_tokentree(nltk_tree):
    if isinstance(nltk_tree, nltk.Tree):
        children = [nltk_tree_to_tokentree(child) for child in nltk_tree]
        return EteTree.TokenTree(nltk_tree.label(), children)
    else:
        return EteTree.TokenTree(nltk_tree.label(), [])