import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from utils import HandlerEllipse

c = mpatches.Circle((0.5, 0.5), 1, facecolor='#f5f5f5', edgecolor='red',
                    linewidth=1)

MOVEMENTS = ['rest', 'power', 'lateral', 'tripod', 'pointer', 'open']
RAW_FS = 2000 # Raw data sampling rate
PROC_FS = 20 # Processed/binned data sampling rate

# Load data
raw_emg_chan_1 = np.loadtxt('data/raw_emg_channel_1.txt')
raw_emg_chan_2 = np.loadtxt('data/raw_emg_channel_2.txt')
pred = np.loadtxt('data/prediction.txt')
pred_proba = np.loadtxt('data/posterior_proba.txt')
state = np.loadtxt('data/control_state.txt')
thresh = np.loadtxt('data/rejection_thresholds.txt')

# Compute time vectors for raw data and predictions
t_raw = np.arange(raw_emg_chan_1.shape[0]) / RAW_FS
t_proc = np.arange(pred.size) / PROC_FS 


# Make the plot
sns.set(rc={'axes.facecolor':'#f5f5f5'}, style="darkgrid", 
            font="Times New Roman", font_scale=0.8)

figsize = (6.9,6)
fig = plt.figure(figsize=figsize)
ax=[]

# First row: threshold selection example
palette = sns.color_palette("Greys", 4)[1:]
lw=2
ax.append(plt.subplot2grid((11, 1), (0, 0), rowspan=2, colspan=1))
ax[0].plot(t_raw, raw_emg_chan_1, color=sns.color_palette("Paired")[8],
  label='EMG channel 1')
ax[0].plot(t_raw, raw_emg_chan_2, color=sns.color_palette("Paired")[9],
  label='EMG channel 2')
ax[0].set_ylabel('EMG')
ax[0].set_xlabel('')
plt.setp(ax[0].get_xticklabels(),visible=False)
plt.setp(ax[0].get_yticklabels(),visible=False)
ax[0].legend(loc=2, bbox_to_anchor=[-0.01, 1.2], frameon=True, edgecolor='.3')

# Second row: prediction vs control
fontsize_leg = 10
lw_prediction = .7
lw_posterior = .7
lw_control = 2.5
ellipse_lw = 1.

ax.append(plt.subplot2grid((11, 1), (2, 0), rowspan=2, colspan=1,
                           sharex=ax[0]))
ax[1].plot(t_proc, pred,label='classification prediction',
  linewidth=lw_prediction, color='k')
ax[1].plot(t_proc, state, label='prosthesis state',
  linewidth=lw_control, color="#3498db")
ax[1].set_yticklabels(MOVEMENTS)
ax[1].set_yticks(np.arange(len(MOVEMENTS)))

handles, labels = ax[1].get_legend_handles_labels()
c = mpatches.Circle((0.3, 0.3), 1, facecolor='#f5f5f5', edgecolor='red',
                    linewidth=1)
leg = ax[1].legend(handles + [c],labels + ['unintended activation'],
                 handler_map={mpatches.Circle: HandlerEllipse()},
                 loc='lower right', bbox_to_anchor=(1., -0.12),
                 frameon=True, edgecolor='.3', framealpha=0.95)

e1 = mpatches.Ellipse((6.9, 1), 3., 0.4,
                     angle=0, linewidth=ellipse_lw, fill=False, zorder=2,
                     color='red')
e2 = mpatches.Ellipse((21.56, 3), 1.7, 0.4,
                     angle=0, linewidth=ellipse_lw, fill=False, zorder=2,
                     color='red')
ax[1].add_patch(e1)
ax[1].add_patch(e2)
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[1].spines['bottom'].set_visible(False)
plt.setp(ax[1].get_xticklabels(),visible=False)

palette = sns.color_palette("Dark2")

# Third row: posterior probabilities
for cc, class_ in enumerate(MOVEMENTS):
    if cc == 0:
        ax.append(plt.subplot2grid((11, 1), (cc+4, 0), rowspan=1, colspan=1,
                                   sharex=ax[1]))
    else:
        ax.append(plt.subplot2grid((11, 1), (cc+4, 0), rowspan=1, colspan=1,
                                   sharex=ax[1], sharey=ax[2]))
    plt.plot(t_proc, pred_proba[:, cc], color = palette[cc],
             label = MOVEMENTS[cc], linewidth=lw_posterior)
    ax[cc+2].set_ylabel(class_)
    ax[cc+2].spines['top'].set_visible(False)
    ax[cc+2].spines['right'].set_visible(False)
    ax[cc+2].set_yticklabels(ax[cc+2].get_yticklabels(), visible=False)
    ax[cc+2].axhline(y=thresh[cc], color='grey', ls='--', lw=0.5)
    ax[cc+2].set_ylim([-0.1,1.1])
    plt.setp(ax[cc+2].get_yticklabels(),visible=False)
    if cc == 4:
        ax[cc+2].text(23.3,0.25, r'$\theta_{}$ = {:.3f}'.format(cc+1,
          thresh[cc]), fontsize=9)
    else:
        ax[cc+2].text(23.3,0.65, r'$\theta_{}$ = {:.3f}'.format(cc+1,
          thresh[cc]), fontsize=9)
    if cc != len(MOVEMENTS)-1:
        plt.setp(ax[cc+2].get_xticklabels(),visible=False)
ax[1].set_xlim((0,26.5))
for ax_ in ax[1:-1]:
    ax_.set_xlabel("")
ax[-1].set_xlabel("Time [s]")
fig.text(0.035, 0.5,"Posterior class probabilities", rotation=90)

fig.text(0.02, 0.97, "a", weight="bold")
fig.text(0.02, 0.81, "b", weight="bold")
fig.text(0.02, 0.61, "c", weight="bold")

fig.tight_layout()
fig.subplots_adjust(hspace=0.2)
plt.savefig('Figure_4.pdf', dpi=600, bbox_inches='tight',
            transparent=False, pad_inches=0)