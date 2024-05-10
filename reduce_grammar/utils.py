import matplotlib.pyplot as plt

from collections import defaultdict
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score
import numpy as np


def plot_results(args, train_accs, dev_accs, test_acc, real_output=False) -> None:
    plt.plot(train_accs)
    plt.plot(dev_accs)
    if test_acc is not None:
        plt.axhline(test_acc, ls='--', lw=2)
    if not real_output:
        plt.ylim(0, 1)
    plt.title("Performance")
    plt.savefig(f"{args.output_dir}/performance_{args.version}_{args.top_k}.png")


