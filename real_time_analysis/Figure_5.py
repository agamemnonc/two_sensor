import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import warnings

from utils import get_early_late_times, get_df_mean_rates, get_df_early_late

warnings.filterwarnings("ignore")

# Load data
df = pd.read_csv('data/experimental_results.csv')
df_mean = get_df_mean_rates(df) # Results with mean completion rates
df_average_time_block_type  = get_df_early_late(df) # Early vs. late
# Get early_times and late_times as numpy arrays
early_times, late_times = get_early_late_times(df_average_time_block_type)

# Make the plot
# Figure properties
sns.set(rc={'axes.facecolor':'#f5f5f5'}, style="darkgrid",
            font="Times New Roman", font_scale=0.8)
figsize=(6.9,6)
fig = plt.figure(figsize=figsize)

# Barplot properties
errwidth = 0.5
linewidth = 1.
ci=95
n_boot=1000
estimator = np.median

# Boxplot properties
box_width = 0.5
box_linewidth = 1.
box_showfliers=False
box_saturation = 0.5
box_fliersize = 3.

# Swarmplot properties
swarm_size = 3
swarm_color = 'k'

# Violinplot properties
violin_cut = 1.

# Colours
palette_subjects = sns.color_palette("Paired", 12)[0:2]
palette_blocks = sns.color_palette("Paired", 12)[2:4]


# First row: completion rate
xticklabels = ["AB " + str(i) for i in range(1,13)] + ["Amp 1", "Amp 2"]
colors = [palette_subjects[0] for i in range(12)] + [
        palette_subjects[1] for i in range(2)]
ax1 = plt.subplot2grid((3, 20), (0, 0), colspan=17)
sns.barplot(data=df_mean, y='Mean completion rate', x="Subject number",
            ax = ax1, palette=colors,
           estimator=estimator, errwidth=errwidth, linewidth=linewidth,
           edgecolor='k',ci=ci, n_boot=n_boot)#, color=colors[0])
ax1.set_ylabel("Completion rate [%]")
ax1.set_xlabel('')
ax1.set_ylim([0, 120])
ax1.set_yticks(np.arange(0,110,10))
sns.set_palette(sns.color_palette("deep"))
ax1.set_xticklabels(xticklabels, rotation=30)
ax2 = plt.subplot2grid((3, 20), (0, 17), colspan=3, sharey=ax1)
sns.boxplot(data=df_mean, y='Mean completion rate', x="Participant",
            ax = ax2, palette=palette_subjects,
           width=box_width, showfliers=box_showfliers, linewidth=box_linewidth)
sns.swarmplot(data=df_mean, y='Mean completion rate', x="Participant",
              ax = ax2,
           size=swarm_size, color=swarm_color)
ax2.set_xlabel('')
ax2.set_yticks(ax1.get_yticks())
ax2.set_ylabel('')
ax2.set_ylim(ax1.get_ylim())
plt.setp(ax2.get_yticklabels(), visible=False)
ax2.set_xticklabels(["AB", "Amp"], rotation=30)

# Second row: completion time
ax3 = plt.subplot2grid((3, 20), (1, 0), colspan=17)
sns.boxplot(data = df[df["Trial_success"]==1], y="Trial_time",
            x="Subject number", ax=ax3, palette=colors,
           width=box_width, showfliers=box_showfliers, linewidth=box_linewidth)
sns.swarmplot(data = df[df["Trial_success"]==1], y="Trial_time",
              x="Subject number", ax=ax3,
             size=swarm_size, color=swarm_color)
ax3.set_ylabel("Completion time [s]")
ax3.set_xlabel("")
xticklabels = ["AB " + str(i) for i in range(1,13)] + ["Amp 1", "Amp 2"]
ax3.set_xticklabels(xticklabels, rotation=30)
ax4 = plt.subplot2grid((3,20), (1,17), colspan=3, sharey=ax3)
sns.violinplot(data = df[df["Trial_success"]==1], y="Trial_time",
               x="Participant", ax=ax4, palette=palette_subjects,
            width=box_width, cut=violin_cut)
plt.setp(ax4.get_yticklabels(), visible=False)
ax4.set_ylabel('')
ax4.set_xlabel('')
xticklabels = ["AB", "Amp"]
ax4.set_xticklabels(xticklabels, rotation=30)
plt.setp(ax4.get_yticklabels(), visible=False)

ax1.set_title("Individual participants")
ax2.set_title("Grouped")

# Row 3: completion time vs. trial number
palette = sns.color_palette("deep", n_colors=2)
estimator = np.median
ax5 = plt.subplot2grid((3,20), (2,0), colspan=17, sharey=ax3)
sns.pointplot(x="Trial", y="Trial_time", hue="Participant",data=df, \
              estimator=estimator, palette=palette_subjects, ax=ax5, dodge=0.1, 
              markers = ["s", "o"], linestyles = ['-', '--'], 
              capsize=.05, ci=95, n_boot=1000)
ax5.set_xticklabels(np.arange(1,11))
ax5.set_xlabel("Trial number")
plt.setp(ax5.collections, sizes=[30])
plt.setp(ax5.lines, linewidth=2.0)
plt.legend(loc='upper right')
ax6 = plt.subplot2grid((3,20), (2,17), colspan=3, sharey=ax3)
sns.boxplot(data=[early_times, late_times],ax=ax6, 
            palette=palette_blocks, width=box_width, linewidth=box_linewidth)
sns.swarmplot(data=[early_times, late_times],ax=ax6, 
            size=swarm_size, color=swarm_color)
ax6.set_yticklabels(ax6.get_yticklabels(), visible=False)
ax6.set_xticklabels(["Early\n(1-2)", "Late\n(9-10)"], rotation=0)
ax5.grid(axis='x')
ax5.axvline(1.5, color='grey', ls='--', lw=0.5)
ax5.axvline(7.5, color='grey', ls='--', lw=0.5)
ax5.legend(loc=2, bbox_to_anchor=[0.48, 0.98], frameon=True, edgecolor='.3')
ax5.text(x=0.16, y=71.5, s='Early')
ax5.text(x=8.2, y=71.5, s='Late')
ax5.set_ylabel('Average completion time [s]')

# statistical annotation
x1, x2 = 0, 1   # columns
y, h, col = 65, 1., 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col,
         fontsize=12)

for ax in [ax3, ax5]:
    ax.set_yticklabels(labels=np.arange(0, 100, 20), visible=True)
    
fig.text(0.02, 0.97, "a", weight="bold")
fig.text(0.02, 0.65, "b", weight="bold")
fig.text(0.02, 0.38, "c", weight="bold")

fig.align_labels()
fig.tight_layout()
fig.subplots_adjust(wspace=0.15, hspace=0.3)

sns.despine(bottom=True, left=True)
plt.savefig('Figure_5.pdf', dpi=600, bbox_inches='tight',
            transparent=False, pad_inches=0)