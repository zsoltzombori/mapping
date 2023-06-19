# visualizing the output of experiments 6, 7, 16, 17, 18, 19, 20, 21 in mlp.py

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('xtick', labelsize=10) 
matplotlib.rc('ytick', labelsize=10)

horizontal=False

fig, ax = plt.subplots()
losses = [ r'Libra-loss', r'Sag-loss', r'NLL-loss', r'$0.75$-merit-loss', r'$0.5$-merit-loss', r'$0.25$-merit-loss', r'uniform-loss', r'RC-loss', r'LWS-loss']
values = [95.5, 99.9, 11.5, 56.6, 64.9, 75.9, 79.7, 41.5, 8.6]


if horizontal:
    ax.barh(losses, width=list(reversed(values)), height=0.3, align='edge')
    ax.set_xlabel(r'Accuracy ($\%$)')
    ax.set_ylabel('Loss function')
    ax.set_xlim([0, 100.0])
    for i in range(10):
        plt.axvline(x=i*10, linewidth=0.1, linestyle='-', color="black", alpha=0.25)
else:
    ax.bar(losses, values)
    ax.set_ylabel(r'Accuracy ($\%$)')
    ax.set_xlabel('Loss function')
    ax.set_ylim([0, 100.0])
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    for i in range(10):
        plt.axhline(y=i*10, linewidth=0.1, linestyle='-', color="black", alpha=0.25)

 

fig.tight_layout()
plt.savefig("small_consistent_dataset.pdf")
