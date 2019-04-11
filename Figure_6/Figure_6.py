import pandas as pd
import warnings
from matplotlib import pyplot as plt
import seaborn as sns

from utils import get_df_early_late
from utils import swarmplot

warnings.filterwarnings("ignore")

df = pd.read_csv('data/raw_emg_power_trial.csv')
df_early_late = get_df_early_late(df)

# Make the plot
# Style options
sns.set(rc={'axes.facecolor':'#f5f5f5'}, style="darkgrid",
            font="Times New Roman", font_scale=0.8)
figsize = (3.45, 2.)

# Boxplot properties
box_width = 0.38
box_fliersize = 3.

# Swarmplot properties
swarm_size = 3
swarm_color = 'k'

# Colour palettes
palette_blocks = sns.color_palette("Paired", 12)[2:4]

fig = plt.figure(figsize=figsize)
ax1 = fig.add_subplot(111)
sns.boxplot(data=df_early_late, hue='Phase', y='Average EMG variance',
            x='Electrodes',
           palette=palette_blocks, fliersize=box_fliersize, width=box_width,
           showfliers=False,
            ax=ax1)
handles, labels = ax1.get_legend_handles_labels()
swarmplot(data=df_early_late, hue='Phase', y='Average EMG variance',
          x='Electrodes',
           dodge=True, color=swarm_color, size=swarm_size, ax=ax1)
ax1.legend(handles=handles, labels=['Early trials (1-2)',
                                    'Late trials (9-10)'],
           bbox_to_anchor=(1., 1.08), edgecolor='.3')
ax1.set_xlabel('Sensor subset')
ax1.set_ylabel('Raw EMG power')
ax1.set_xticklabels(['Used', 'Non-used'])
# statistical annotation
x1, x2 = -0.1, 0.1   # columns
y, h, col = 1.3e-8, 3e-10, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+2*h, "n.s.", ha='center', va='bottom', color=col)

x1, x2 = 0.9, 1.1   # columns
y, h, col = 0.85e-8, 2e-10, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+0.8*h, "*", ha='center', va='bottom', color=col,
         fontsize=12)
plt.savefig('Figure_6.pdf', dpi=600, bbox_inches='tight',
            transparent=False, pad_inches=0)