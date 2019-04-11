import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import warnings

warnings.filterwarnings("ignore")

# Load data
df = pd.read_csv('data/average_results.csv')

# Fit robust regression models between offline metrics (classification
# accuracy and logarithmic loss) and average completion time.

X = sm.add_constant(df["Balanced cross-entropy loss"])
y = df["Average completion time"]
est_cel = sm.RLM(y, X)
est_cel = est_cel.fit()
pvalue_cel = est_cel.pvalues[1]

X = sm.add_constant(df["Balanced classification accuracy"])
y = df["Average completion time"]
est_ca = sm.RLM(y, X)
est_ca = est_ca.fit()
pvalue_ca = est_ca.pvalues[1]

# Make the plot
sns.set(rc={'axes.facecolor':'#f5f5f5'}, style="darkgrid",
            font="Times New Roman", font_scale=0.8)
figsize=(3.45,4.5)
palette_subjects = sns.color_palette("Paired", 12)[0:2]

is_robust = True
n_boot = 1000
ci = 95
truncate = False
scatter_size = 20
lw = 2

fig, (ax1, ax2) = plt.subplots(2,1, figsize=figsize, sharey=True)
sns.regplot(x="Balanced cross-entropy loss", 
            y = "Average completion time", 
            data = df,
            n_boot = n_boot,
            ci = ci,
            robust=is_robust, 
            truncate = truncate,
            color = 'grey',
            scatter_kws = {"s" : scatter_size},
            line_kws = {"lw" : lw},
            ax=ax1)
ax1.scatter(
    df[df["Participant"]=="Able-bodied"]["Balanced cross-entropy loss"].values,
    df[df["Participant"]=="Able-bodied"]["Average completion time"].values, 
    color=palette_subjects[0],
    marker='s',
    s=scatter_size,
    label="Able-bodied")
ax1.scatter(
    df[df["Participant"]=="Amputee"]["Balanced cross-entropy loss"].values,
    df[df["Participant"]=="Amputee"]["Average completion time"].values, 
    color=palette_subjects[1],
    s=scatter_size,
    label="Amputee")
ax1.annotate("p = {:.3f}".format(pvalue_cel), 
             xy = (0.77, 27), bbox=dict(facecolor='none', edgecolor='.8'))
ax1.set_ylabel("Completion time [s]")
ax1.set_xlabel('Cross-entropy loss')
ax1.set_xlim(right=0.9)
ax1.legend(loc=2, bbox_to_anchor=(0.4, 1.02), ncol=1, frameon=True,
           edgecolor='.3')
sns.regplot(x="Balanced classification accuracy", 
            y = "Average completion time", 
            data = df,
            n_boot = n_boot,
            ci = ci,
            robust=is_robust, 
            truncate = truncate,
            color = 'grey',
            scatter_kws = {"s" : scatter_size},
            line_kws = {"lw" : lw},
            ax=ax2)

ax2.scatter(
    df[df["Participant"]=="Able-bodied"][
            "Balanced classification accuracy"].values,
    df[df["Participant"]=="Able-bodied"]["Average completion time"].values, 
    color=palette_subjects[0],
    s = scatter_size,
    marker='s',
    label="Able-bodied")
ax2.scatter(
    df[df["Participant"]=="Amputee"][
            "Balanced classification accuracy"].values,
    df[df["Participant"]=="Amputee"]["Average completion time"].values, 
    color=palette_subjects[1],
    s = scatter_size,
    label="Amputee")
ax2.set_ylabel("Completion time [s]")
ax2.set_xlabel('Classification accuracy')
ax2.annotate("p = {:.3f}".format(pvalue_ca), 
             xy = (71.7, 27), bbox=dict(facecolor='none', edgecolor='.8'))

fig.text(0.04, 0.96, "a", weight="bold")
fig.text(0.05, 0.48, "b", weight="bold")
fig.tight_layout()
fig.subplots_adjust(wspace=0.1)
plt.savefig('Figure_7.pdf', dpi=600, bbox_inches='tight',
            transparent=False, pad_inches=0)