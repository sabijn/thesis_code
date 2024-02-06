# Code for creating a chart-based probing experiment


## Notes per file
### `data.py`

```python
def tree_to_spanlabels(tree, device, merge_pos=False, merge_at=False, skip_labels=None):
    ...
    tree.treeposition_spanning_leaves(i, j)
    ...
```

The treepositions method returns a list of the tree positions of subtrees and leaves in a tree. By default, it gives the position of every tree, subtree, and leaf, in prefix order.
The treeposition_spanning_leaves returns the position of the nodes that cover the range of leaves from index i to index j in the tree.


## Notes
1. The already generated data is generated on CPU.  
If you want to use it on cuda or mps, you should delete the all the data in the data directory  
and run the experiment. The data will be created automtically on the right device.

2. If you get bored you can:  
    - Write docstrings
    - Write a dataloader instead of the handmade one
    - Write comments in general

3. For the probe classifier:  
    - Maybe insert 'not a constituent' also in the data
    - Maybe not and fill all not present spans in the matrix with 0 (first go with this one, if it does not work try something else)

