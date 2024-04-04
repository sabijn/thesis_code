import matplotlib.pyplot as plt

def plot_results(train_accs, dev_accs, test_acc, real_output=False):
    plt.plot(train_accs)
    plt.plot(dev_accs)
    if test_acc is not None:
        plt.axhline(test_acc, ls='--', lw=2)
    if not real_output:
        plt.ylim(0, 1)
    plt.title("Performance")
    plt.show()