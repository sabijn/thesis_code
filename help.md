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

## Poetry
You switched from conda to poetry 
How to create a .toml file
```shell
poetry init
```

After creating your project: check if the project is installed in your directory (check for an .env folder).  

If you manually added packages to the dependencies file
```shell
poetry install
```

Add dependencies from command line
```shell
poetry add <package_name>
```

Open shell
```shell
poetry shell
```

Info on your environment
```shell
poetry env info 
```
```-p``` print only the path of your environment  

Show all installed packages
```shell
poetry show
```

## Conda
Listing environment variables per environment
```shell
conda env config vars list -n myenv
```
Setting environment variables
```shell
conda env config vars set MY_VAR=something OTHER_THING=ohhhhya
```

Deleting environment variables 
```shell
conda env config vars unset MY_VAR
```

## GIT
Remove files from version control but leaves the local copy
```shell
git rm -r -f --cached DirectoryName
```
-r: recursively  
-f: force, if you have doubts, just do it
--cached: leave the local copies!  

### Branches
**Create** branch
```shell
git branch <name_new_branch>
```
or 
```shell
git checkout -b <name_new_branch>
```

**Switch** to branch
```shell
git checkout <branch_name>
```

**Merge** with master
```shell
git checkout master
git merge <branch_name>
```

**Delete** branch
```shell
git branch -d <branch_name>
```

**Info** on your current branch
```shell
git status
```

List branches
```shell
git branch
```

**Merging**:
1. 
From branch: 
```shell
git merge master
```
2. Usually you merge from the branch you want to merge into:
```shell
git checkout master
git merge <branch_name>
```

You know have a branch that does not an upstream equivalent. That means that you cannot push to your remote from this branch.
You can only commit the changes and than merge with main. Main can than push it to your remote repository.

**Create upstream**
```shell
git push --set-upstream origin <branch_name> 
```
When on the branch

## Explanation per directory

### data
- activations_notconcat.pickle: sentence activations in the form of a dictionary with as key a layer and as value a list containing sentence representations per word. These sentence representations are torch tensors of the shape (amount of words in sentence, embedding dim)
- activations.pickle: sentence activations in the form of a dictionary with as key a layer and as value a torch tensor with stacked sentence representations. Only use if you need sentence representations instead of individual word representations
- activations_combined_avg.pickle: activations for LCA prediction. Word pair activations are averaged.
- activations_combined_concat.pickle: activations for LCA prediction. Word pair activations are concatenated with dim=1
- activations_combined_max.pickle: activations for LCA prediction. Word pair activations are based on the max representation of the word pair.
- activations_notconcat.pickle: activations per word pair concatenated over layers 3, 6 and 8. The structure is a dictionary with one key (0) with as value a list with representations per sentence (tensor with the shape of the amount of word pairs). Dictionary structure to keep the code in main.py modular.
- train_bies_labels.txt : labels for chunking task
- train_rel_labels.txt : tuple tree representation of LCA's
- train_rel_toks.txt : label of LCA's
- train_shared_levels.txt : relative depth of LCA's
- train_text_bies.txt : train sentences
- train_text.txt : train sentences. Equal to train_text_bies
- train_unaries.txt : labels of single consistuent phrases. 
- combined_predictions.txt : contains per word a tuple containing LCA_SHARED_UNARY
- sentences_postags.pickle : list of lists containing tuples (word, postag)

## Terminal colors
1. conda config --set changeps1 false
2. Edited prompt line in ~/.bash_profile