from pathlib import Path
from collections import Counter

home_path = Path("/Users/sperdijk/Documents/Master/Jaar 3/Thesis/thesis_code")

# read in train_shared_levels.txt
shared_levels = []
with open(home_path / Path('data/train_shared_levels.txt')) as f:
    # shared levels
    for l in f:
        shared_levels.extend(l.split(' '))

# count per value in shared_levels
print(Counter(shared_levels))
