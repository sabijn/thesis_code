import matplotlib.pyplot as plt
# print(plt.style.available)

# open results file
results_file = open('results/results_lca_avg.txt', 'r')
results_avg = results_file.readlines()
results_file.close()

results_file = open('results/results_lca_max.txt', 'r')
results_max = results_file.readlines()
results_file.close()

results_file = open('results/results_lca_concat.txt', 'r')
results_concat = results_file.readlines()
results_file.close()

all_test = []
all_val = []
for results in [results_avg, results_max, results_concat]:
    test_accs = []
    val_accs = []
    for line in results:
        line = line.strip()
        # get test and val acurracy from string with this format: " {'test': 0.8542600870132446, 'val': 0.8590031266212463}\n"
        if line.startswith('{'):
            test_acc = float(line.split(',')[0].split(':')[1].split('}')[0])
            val_acc = float(line.split(',')[1].split(':')[1].split('}')[0])
            test_accs.append(test_acc)
            val_accs.append(val_acc)
    all_test.append(test_accs)
    all_val.append(val_accs)

color = ['C1', 'C2', 'C0']
i = 0
for test_accs, val_accs, name in zip(all_test, all_val, ['avg', 'max', 'concat']):
    plt.plot(test_accs, label=f'test {name}', color=color[i])
    plt.plot(val_accs, label=f'val {name}', linestyle='--', color=color[i])   
    i += 1

# plt.plot(test_accs, label='test')
# plt.plot(val_accs, label='val')
plt.legend()
plt.xlabel('Layer')
plt.ylabel('Accuracy')
plt.title('Predicting LCA')
plt.savefig('results/lca_per_layer.png')
