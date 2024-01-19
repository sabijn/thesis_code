import matplotlib.pyplot as plt
import numpy as np
import os

# Helper function to plot confusion matrix
def plot_confusion_matrix(cm, swapped_vocab):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

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
            ax.text(j, i, str(round(cm[i, j], 2)),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(os.environ['CURRENT_WDIR'] + '/results/shared_levels/confusion_matrix.png')

import numpy as np

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
    home = os.environ['CURRENT_WDIR']
    cm = np.load(home + '/results/shared_levels/confusion_matrix_0.npy')
    label_vocab = {int('1'): 0, int('-3'): 1, int('4'): 2, int('0'): 3, int('3'): 4, int('-4'): 5, int('2'): 6, int('-2'): 7, int('-1'): 8}
    sorted_vocab = dict(sorted(label_vocab.items()))
    swapped_vocab = {v: k for k, v in sorted_vocab.items()}
    rcm = reorder_confusion_matrix(cm, sorted_vocab.values())
    plot_confusion_matrix(rcm, swapped_vocab)