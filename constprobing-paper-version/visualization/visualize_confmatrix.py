import matplotlib.pyplot as plt
import numpy as np
import os

# Helper function to plot confusion matrix
def plot_confusion_matrix(cm, swapped_vocab):
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    # ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=swapped_vocab.values(), yticklabels=swapped_vocab.values(),
           title='Confusion Matrix: relative shared levels',
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(round(cm[i, j], 2)) if cm[i, j] != 0 else '0',
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig('/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code' + '/results/shared_levels/confusion_matrix.png')


def reorder_confusion_matrix(conf_matrix, desired_order):
    """
    Reorder the confusion matrix based on the desired label order.

    :param conf_matrix: The original confusion matrix as a 2D numpy array.
    :param desired_order: A list representing the desired order of labels.
    :return: Reordered confusion matrix as a 2D numpy array.
    """
    # Create a new confusion matrix with the same size but with reordered labels
    reordered_matrix = np.zeros_like(conf_matrix)

    # Fill in the reordered matrix
    for i, orig_i in enumerate(desired_order):
        for j, orig_j in enumerate(desired_order):
            reordered_matrix[i, j] = conf_matrix[orig_i, orig_j]

    return reordered_matrix

if __name__ == '__main__':
    home = '/Users/sperdijk/Documents/Master/Jaar_3/Thesis/thesis_code'
    cm = np.load(home + '/results/shared_levels/confusion_matrix_0.npy')
    label_vocab = {'-3': 0, '-9': 1, '9': 2, '-6': 3, '7': 4, '8': 5, '-8': 6, '5': 7, '-13': 8, '-12': 9, '-14': 10, '-15': 11, '-2': 12, '1': 13, '11': 14, '-4': 15, '-7': 16, '-10': 17, '2': 18, '-1': 19, '3': 20, '-5': 21, '6': 22, '10': 23, '4': 24, '-11': 25, 'ROOT': 26}

    swapped_vocab = {v: k for k, v in label_vocab.items()}
    for k, v in swapped_vocab.items():
        if v != 'ROOT':
            swapped_vocab[k] = int(v)
        else:
            swapped_vocab[k] = 12
    
    # sort on value
    sorted_vocab = {k: v for k, v in sorted(swapped_vocab.items(), key=lambda item: item[1])}
    print(sorted_vocab)
    rcm = reorder_confusion_matrix(cm, sorted_vocab.values())
    sorted_vocab[26] = 'ROOT'
    plot_confusion_matrix(rcm, sorted_vocab)