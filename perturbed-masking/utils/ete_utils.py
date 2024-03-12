from ete3 import Tree as EteTree
import torch
from tqdm import tqdm
import nltk
import re
import logging

logger = logging.getLogger(__name__)


class FancyTree(EteTree):
    def __init__(self, *args, **kwargs):
        self.word2idx = kwargs.pop('word2idx', None)
        self.idx2word = kwargs.pop('idx2word', None)
        
        super().__init__(*args, format=1, **kwargs)
        
    def __str__(self):
        return self.get_ascii(show_internal=True)
    
    def __repr__(self):
        return str(self)

def create_ete3_from_pred(sentence):
    tree_idx = 1
    newick_str = ''
    split_sentence = sentence.split(' ')

    for i, element in enumerate(split_sentence):
        if element == '(':
            newick_str += element
        elif element == ')':
            newick_str += element + str(tree_idx)
            tree_idx += 1
        else:
            newick_str += str(tree_idx)
            tree_idx += 1
        
        if (i != len(split_sentence) - 1 and 
            split_sentence[i + 1] != ')' and element != '('):
            newick_str += ','

    return FancyTree(f"{newick_str};")

def _remove_postags(input_string):
    # Pattern to match '<capital letter>_<number>' and optional brackets if directly before a word
    #pattern = r'\(?(?=[A-Z]*_\d+\) )?[A-Z]*_\d+\)?\s?'
    pattern = r'[A-Z]*_[0-9]*'
    # Replace the pattern with an empty string
    output_string = re.sub(pattern, ' ', input_string)

    return output_string

def _add_space(input_string):
    new_string = ''
    split_string = input_string.split(' ')

    for i, element in enumerate(split_string):
        if re.search("[a-zA-Z]*\)", element):
            split_text = re.split(r'(?<=[a-zA-Z.])(?=\))', element)

            new_string += split_text[0] + ' ' + ' '.join([e for e in split_text[1]]) + ' '
        elif element == ' ' or element == '':
            continue
        else:
            new_string += f'{element} '

    return new_string.rstrip(' ')

def _simplify_tree(node):
    """
    Simplify the tree by removing nodes that have only one child and connecting
    the child directly to the node's parent.
    """
    # List to keep track of nodes to be removed
    nodes_to_remove = []

    # Traverse tree in post-order: children before their parents
    for child in node.traverse("postorder"):
        # Check if the node has exactly one child
        if len(child.children) == 1:
            nodes_to_remove.append(child)

    # Remove identified nodes and connect their children to the nodes' parents
    for node_to_remove in nodes_to_remove:
        parent = node_to_remove.up  # Get the parent of the node to be removed
        child = node_to_remove.children[0]  # Get the single child
        if parent:  # If the node to remove is not the root
            # Connect the child directly to the parent of the node to be removed
            parent.add_child(child)
            # Remove the node from its parent
            node_to_remove.detach()
        else:
            # If the node to remove is the root and has only one child, update the tree's root
            node.set_outgroup(child)
    
    return node

def _reset_node_names(tree):
    """
    Reset the names of all nodes in the tree based on their preorder traversal index.
    """
    # Traverse tree in preorder and update names
    for index, node in enumerate(tree.traverse("postorder")):
        node.name = str(index + 1)
    
    return tree

def gold_tree_to_ete(gold_tree):
    # Remove the postags from the gold tree
    gold_tree = _remove_postags(gold_tree)
    gold_tree = _add_space(gold_tree)

    # Create the ete3 tree
    gold_tree = create_ete3_from_pred(gold_tree)
    print(gold_tree)
    gold_tree = _simplify_tree(gold_tree)
    gold_tree = _reset_node_names(gold_tree)

    return gold_tree


def create_distances(corpus):
    all_distances = []

    for ete_tree in tqdm(corpus):
        sen_len = len(ete_tree.search_nodes())
        distances = torch.zeros((sen_len, sen_len))

        for node1 in ete_tree.search_nodes():
            for node2 in ete_tree.search_nodes():
                distances[int(node1.name)-1][int(node2.name)-1] = node1.get_distance(node2)

        all_distances.append(distances)

    return all_distances