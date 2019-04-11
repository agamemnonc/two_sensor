import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import warnings

from utils import load_results, load_confusion_matrices

warnings.filterwarnings("ignore")

# Load results and confusion matrices
results_df = load_results('data/offline_analysis_results.csv')
cm_ab_mean, cm_amp_mean, labels = load_confusion_matrices(
        'data/confusion_matrices_all.npy', classifier='RDA', n_sensors=2)

# Make the plot
sns.set(rc={'axes.facecolor':'#f5f5f5'}, style="darkgrid",
            font="Times New Roman", font_scale=0.8)
figsize=(6.9,5.5)

# First row: cross-entropy for varying number of sensors
errwidth=2
dodge=0.2
estimator=np.median
markers = ["s", "o", "d"]
linestyles = ['-', '--', '-.']
point_size = 8
capsize=.1
ci = 95
n_boot = 1000
hue_order = ["LDA", "RDA", "QDA"]
linewidth = 1.
point_scale = 1.

# Swarmplot properties
swarm_size = 3
swarm_color = 'k'

# Boxplot properties
box_width = 0.25
box_linewidth = 1.
box_showfliers=False
box_saturation = 0.5
box_fliersize = 3.

palette = sns.color_palette("Reds", 6)[1::2]

fig = plt.figure(figsize=figsize)
ax1 = fig.add_subplot(321)
sns.pointplot(x="number of sensors", y="logloss", hue="Classifier",
              data=results_df[results_df.participant=="Able-bodied"], 
              estimator=estimator, ax=ax1, dodge=dodge, n_boot=n_boot,
              hue_order = hue_order, capsize=capsize, errwidth=errwidth,
              ci=ci, markers = markers, 
              linestyles = linestyles, scale=point_scale, palette=palette)
plt.setp(ax1.collections, sizes=[point_size])
plt.setp(ax1.lines, linewidth=linewidth)
ax1.legend(loc=2, ncol=2, frameon=True, edgecolor='.3',
           bbox_to_anchor=(0., 1.03))
ax1.set_title("Able-bodied (n=12)")
ax1.set_ylabel('Cross-entropy loss')
ax1.set_xlabel('Number of added sensors')
ax1.xaxis.set_ticks(np.arange(0, 16, 4))
ax1.xaxis.set_ticklabels(np.arange(1, 17, 4))
ax1.grid(axis='x')
ax2 = fig.add_subplot(322, sharey=ax1)
sns.pointplot(x="number of sensors", y="logloss", hue="Classifier",
              data=results_df[results_df.participant=="Amputee"], 
              estimator=estimator, ax=ax2, dodge=dodge, n_boot=n_boot,
             hue_order = hue_order, capsize=capsize, errwidth=errwidth,
             ci=ci, markers = markers, 
              linestyles = linestyles, scale=point_scale, palette=palette)
plt.setp(ax2.get_yticklabels(), visible=False)
plt.setp(ax2.collections, sizes=[point_size])
plt.setp(ax2.lines, linewidth=linewidth)
ax2.legend_.remove()
ax2.set_title("Amputee (n=2)")
ax2.set_xlabel('Number of added sensors')
ax2.set_ylabel('')
ax2.xaxis.set_ticks(np.arange(0, 16, 4))
ax2.xaxis.set_ticklabels(np.arange(1, 17, 4))
ax2.grid(axis='x')


# Second row: Cross-entropy boxplots for two sensors
ax3 = fig.add_subplot(323)
sns.boxplot(data=results_df[(results_df["number of sensors"]==2) &
                            (results_df["participant"]=="Able-bodied")],
x='Classifier', y='logloss', 
            ax = ax3, width=box_width, linewidth=box_linewidth,
            palette=palette, showfliers=box_showfliers)
sns.swarmplot(data=results_df[(results_df["number of sensors"]==2) &
                              (results_df["participant"]=="Able-bodied")],
x='Classifier', y='logloss', 
            ax = ax3, size=swarm_size, color=swarm_color)
ax4 = fig.add_subplot(324, sharey=ax3)
sns.boxplot(data=results_df[(results_df["number of sensors"]==2) &
                            (results_df["participant"]=="Amputee")],
x='Classifier', y='logloss', 
            ax = ax4, width=box_width, linewidth=box_linewidth,
            palette=palette, showfliers=box_showfliers)
sns.swarmplot(data=results_df[(results_df["number of sensors"]==2) &
                              (results_df["participant"]=="Amputee")],
x='Classifier', y='logloss', 
            ax = ax4, size=swarm_size, color=swarm_color)
plt.setp(ax4.get_yticklabels(), visible=False)
ax3.set_ylabel('Cross-entropy loss')
ax3.set_ylim([0.2, 1.2]) # Hide one outlier
ax4.set_ylabel(" ")
ax3.set_xlabel("Classifier")
ax4.set_xlabel("Classifier")
plt.setp(ax3.get_yticklabels(), visible=True)

# Third row: Confusion matrices
# Hetmap properties
cmap='PuRd'
vmin = 0
vmax = 100
linewidths = 0.5

ax5 = fig.add_subplot(325)
ax6 = fig.add_subplot(326, sharey=ax5, sharex=ax5)
cbar_axis = fig.add_axes([0.17, 0., 0.76, .02])

for ax, cm_tmp, cbar_ax in zip([ax5, ax6], [cm_ab_mean, cm_amp_mean],
                               [cbar_axis, None]):
    cm_tmp_norm =  cm_tmp/cm_tmp.sum(axis=1)[:, np.newaxis]
    sns.heatmap(100*cm_tmp_norm,  cmap=cmap, vmin=vmin, vmax=vmax,
                linewidths=linewidths, 
                ax = ax, cbar=True if cbar_ax else False, cbar_ax=cbar_axis,
                cbar_kws={"orientation": "horizontal"},
                annot=True, fmt = "3.1f",
               annot_kws = {"fontsize": 10})
    ax.set_xticklabels(labels, rotation=0)
    ax.set_yticklabels(labels, rotation=0)
    ax.set_xlabel('Predicted class')
    plt.setp(ax.get_yticklabels(), visible=False)
plt.setp(ax5.get_yticklabels(), visible=True)
ax5.set_ylabel('True class')
ax6.set_ylabel('')
fig.text(0.02, 0.96, "a", weight="bold")
fig.text(0.02, 0.67, "b", weight="bold")
fig.text(0.02, 0.36, "c", weight="bold")

fig.align_ylabels(axs=[ax1, ax3, ax5])
fig.tight_layout(rect=[0, 0.02, 1, 1])
plt.subplots_adjust(wspace=0.02, hspace=0.4)
plt.savefig('Fig_3.pdf', dpi=600, bbox_inches='tight',transparent=False,
            pad_inches=0)