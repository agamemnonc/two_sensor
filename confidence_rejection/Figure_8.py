import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as 
import warnings

warnings.filterwarnings("ignore")

# Cutoff for False positive rate
FPR_CUTOFF = 0.0005

# Load data for shown participant and movement class
roc_data = pd.read_csv('data/roc_lateral.csv')

sns.set(rc={'axes.facecolor':'#f5f5f5'}, style="darkgrid", font="Times New Roman", font_scale=0.8)

movement = 2

figsize = (3.45,4.5)
fig = plt.figure(figsize=figsize)

# First row: threshold selection example
palette = sns.color_palette("Greys", 4)[1:]
thresh =  roc_data.Thresholds[np.where(roc_data.FPR>FPR_CUTOFF)[0][0]]
lw=2
ax1 = fig.add_subplot(2,1,1)
ax1.plot([-0.015, -0.015, 1], [0, 1.005, 1.005], color=palette[1], lw=lw, linestyle='-.', label='Perfect')
ax1.plot(roc_data.FPR, roc_data.TPR,lw=lw, color=palette[2], label='RDA')
ax1.plot([0, 1], [0, 1], color=palette[0], lw=lw, linestyle='--', label='Random')

ax1.set_xlim([-0.1, 1.01])
ax1.set_ylim([0, 1.05])
ax1.set_xlabel("False positive rate")
ax1.set_ylabel("True positive rate")
#ax[0].legend(bbox_to_anchor=[0.75, 0.98])
ax1.legend(loc='lower right', frameon=True, edgecolor=".3", bbox_to_anchor=(0.99, 0.0))
#ax[0].grid()
ax1.set_title("ROC (original scale)")

ax2 = fig.add_subplot(2,1,2)
ax2.plot(roc_data.FPR, roc_data.TPR, label='RDA classifier', color=palette[2], lw=lw)
ax2.set_xlim([-0.0001, 0.002])
ax2.set_xlabel("False positive rate")
ax2.set_ylabel("True positive rate")
ax2.vlines(x=FPR_CUTOFF, ymin=0, ymax=1, colors='grey', linestyles='--', color=palette[0], label='FPR cut-off')
ax2.annotate(r'$\theta_c $ = {:.3f}'.format(thresh), 
             xy=(FPR_CUTOFF, roc_data.TPR[np.where(roc_data.FPR>FPR_CUTOFF)[0][0]]), 
             xytext=(FPR_CUTOFF+3*1e-4, 0.25),
             arrowprops=dict(facecolor='black', color='k', shrink=0.01, width=1, headwidth=4, headlength=4),)
ax2.legend(loc='upper right', bbox_to_anchor=[0.78, 1.025],
            frameon=True, edgecolor='.3', framealpha=0.95)
ax2.set_title("ROC (x-axis zoom)")
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

fig.text(0.02, 0.95, "a", weight="bold")
fig.text(0.02, 0.5, "b", weight="bold")

fig.tight_layout()
plt.savefig('Figure_8.pdf', dpi=600, bbox_inches='tight',
            transparent=False, pad_inches=0)