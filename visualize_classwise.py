# import pickle
# import matplotlib.pyplot as plt

# label_vocab = {'I': 0, 'E': 1, 'S': 2, 'B': 3, 'PCT': 4}

# # read test_final.pickle
# with open('test_final.pickle', 'rb') as f:
#     test_final = pickle.load(f)

# # read val_final.pickle
# with open('val_final.pickle', 'rb') as f:
#     val_final = pickle.load(f)

# class0 = []
# class1 = []
# class2 = []
# class3 = []
# class4 = []

# for layer in val_final:
#     for c in layer[0].keys():
#         if c == 'test_acc':
#             continue
            
#         if c == 'class_0':
#             class0.append(layer[0][c])
#         elif c == 'class_1':
#             class1.append(layer[0][c])
#         elif c == 'class_2':
#             class2.append(layer[0][c])
#         elif c == 'class_3':
#             class3.append(layer[0][c])
#         elif c == 'class_4':
#             class4.append(layer[0][c])

# # plot all classes in the same lot
# plt.plot(class0, label=f'class I')
# plt.plot(class1, label=f'class E')
# plt.plot(class2, label=f'class S')
# plt.plot(class3, label=f'class B')
# plt.plot(class4, label=f'class PCT')
# plt.legend()
# plt.xlabel('Layer')
# plt.ylabel('Accuracy')
# plt.title('Accuracy per class for chunking task (valset).')
# plt.savefig('results/chunking_class_valset.png')

import pickle
import matplotlib.pyplot as plt

label_vocab = {'I': 0, 'E': 1, 'S': 2, 'B': 3, 'PCT': 4}

# read test_final.pickle
with open('test_final.pickle', 'rb') as f:
    test_final = pickle.load(f)

# read val_final.pickle
with open('val_final.pickle', 'rb') as f:
    val_final = pickle.load(f)

class0 = []
class1 = []

for layer in val_final:
    for c in layer[0].keys():
        if c == 'test_acc':
            continue
            
        if c == 'class_0':
            class0.append(layer[0][c])
        elif c == 'class_1':
            class1.append(layer[0][c])

# plot all classes in the same lot
plt.plot(class0, label=f'class I')
plt.plot(class1, label=f'class E')
plt.legend()
plt.xlabel('Layer')
plt.ylabel('Accuracy')
plt.title('Accuracy per class for chunking task (valset).')
plt.savefig('results/chunking_class_valset.png')