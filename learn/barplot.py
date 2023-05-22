# visualizing the output of experiments 6, 7, 16, 17, 18, 19, 20, 21 in mlp.py

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('xtick', labelsize=10) 
matplotlib.rc('ytick', labelsize=10)

fig, ax = plt.subplots()
losses = [ r'Libra-loss', r'NLL-loss', r'$0.75$-merit-loss', r'$0.5$-merit-loss', r'$0.25$-merit-loss', r'uniform-loss', r'RC-loss']
values = [95.5, 11.5, 56.6, 64.9, 75.9, 79.7, 41.5]
ax.bar(losses, values)
ax.set_ylabel(r'Accuracy ($\%$')
ax.set_xlabel('Loss function')
ax.set_ylim([0, 100.0])
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
fig.tight_layout()
plt.savefig("small_consistent_dataset.pdf")
