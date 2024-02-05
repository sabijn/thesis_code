import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np 

home_path = Path("/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code")

# 5000 Layer 0 {'test': [{'test_acc': 0.6850000023841858, 'class_0': 0.5384615659713745, 'class_1': 0.0, 'class_2': 0.7777777910232544, 'class_3': 0.0, 'class_4': 0.6000000238418579, 'class_5': 0.0, 'class_6': 1.0, 'class_7': 0.5, 'class_8': 0.0}], 'val': [{'test_acc': 0.6713333129882812, 'class_0': 1.0, 'class_1': 0.0, 'class_2': 0.8947368264198303, 'class_3': 0.0, 'class_4': 0.5, 'class_5': 0.0, 'class_6': 0.6499999761581421, 'class_7': 0.25, 'class_8': 0.0}]}
# 10000  {'test': [{'test_acc': 0.6809999942779541, 'class_0': 0.8333333134651184, 'class_1': 0.0, 'class_2': 0.0, 'class_3': 1.0, 'class_4': 0.0, 'class_5': 0.0, 'class_6': 0.6666666865348816, 'class_7': 1.0, 'class_8': 0.75}], 'val': [{'test_acc': 0.671999990940094, 'class_0': 0.8333333134651184, 'class_1': 0.0, 'class_2': 0.0, 'class_3': 0.8235294222831726, 'class_4': 0.0, 'class_5': 0.0, 'class_6': 1.0, 'class_7': 1.0, 'class_8': 0.3333333432674408}]}


#

# label_vocab_5000 = {'2': 0, '-2': 1, '-1': 2, '3': 3, '-4': 4, '0': 5, '1': 6, '4': 7, '-3': 8}
# label_vocab_10000 = {'1': 0, '-3': 1, '-2': 2, '-1': 3, '0': 4, '3': 5, '2': 6, '-4': 7, '4': 8}

# # sort dicts on keys
# label_vocab_5000 = dict(sorted(label_vocab_5000.items()))
# label_vocab_10000 = dict(sorted(label_vocab_10000.items()))
# print(label_vocab_5000)
# # {'-1': 2, '-2': 1, '-3': 8, '-4': 4, '0': 5, '1': 6, '2': 0, '3': 3, '4': 7}
# print(label_vocab_10000)
# # {'-1': 3, '-2': 2, '-3': 1, '-4': 7, '0': 4, '1': 0, '2': 6, '3': 5, '4': 8}


 
# # set width of bar 
barWidth = 0.4
fig = plt.subplots(figsize =(12, 8)) 
 
# # set height of bar 
results_5 = [0.6000000238418579, 0.0, 0.0, 0.7777777910232544, 0.0, 1.0, 0.5384615659713745, 0.0, 0.5]
results_10 = [1.0, 0.0, 0.0, 1.0, 0.0, 0.8333333134651184, 0.6666666865348816, 0.0, 0.3333333432674408]
 
# # Set position of bar on X axis 
br1 = np.arange(len(results_5)) 
br2 = [x + barWidth for x in br1] 
 
# # Make the plot
plt.bar(br1, results_5, color ='r', width = barWidth, 
        edgecolor ='grey', label ='sampling=5000') 
plt.bar(br2, results_10, color ='g', width = barWidth, 
        edgecolor ='grey', label ='sampling=10000') 
 
# # Adding Xticks 
plt.xlabel('Amount of sampling', fontweight ='bold', fontsize = 15) 
plt.ylabel('Accuracy', fontweight ='bold', fontsize = 15) 
plt.xticks([r + (barWidth / 2) for r in range(len(results_5))], 
        ['-4', '-3', '-2', '-1', '0', '1', '2', '3', '4'])
plt.title('Accuracy per class for different sample sizes.')
 
plt.legend()
plt.savefig(home_path / 'results' / 'classwise_sampling.png')