# Notes concerning this project

## Code notes

*Continue* skips the remaining code in the loop and *continues* to the next iteration.  
For example, the following code only prints 2, 3, 4, 6, 8, 9.
```python

for k, num in enumerate(range(10)):
    if k in [0, 1, 5, 7]:
        continue
    print(k)
```

*Break* breaks out of the entire for loop.  
For example, the following code does not print anything.
```python

for k, num in enumerate(range(10)):
    if k in [0, 1, 5, 7]:
        break
    print(k)
```

## NLTK

Tree.leaf_treeposition()
```python
tree.leaf_treeposition(i)
```
Returns: tuple representation of tree position  
To convert this representation to a label:
```python
tree[tree.leaf_treeposition(i)]
```
For non-terminal nodes, converting to a label:
```python
lowest_common_ancestor = tree.treeposition_spanning_leaves(i,j+1)
tree[lowest_common_ancestor].label()
```
