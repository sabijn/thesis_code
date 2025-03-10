from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import os

home_path = os.environ['CURRENT_WDIR']

# read in train_shared_levels.txt
shared_levels = []
with open(home_path / Path('data/train_shared_withoutROOT.txt')) as f:
    # shared levels
    for l in f:
        shared_levels.extend(l.strip('\n').split(' '))

# count per value in shared_levels
data = Counter(shared_levels)
print(data)
# short data based on keys
data = {k: v for k, v in sorted(data.items(), key=lambda item: int(item[0]))}
plt.bar(list(data.keys()), list(data.values()))
# plot x label in 45 degree angle
plt.xticks(rotation=45)
plt.xlabel('Shared levels')
plt.ylabel('Frequency')
plt.title('Distribution of shared levels')
plt.savefig(home_path / Path('results/distribution_shared_levels_balanced.png'))

