import pandas as pd
from itertools import combinations
from scipy.stats import wilcoxon, friedmanchisquare

from utils import load_results

results_df = load_results('data/offline_analysis_results.csv')

lda_logloss = results_df[(results_df["number of sensors"]==2) & (results_df["Classifier"] == "LDA")].logloss.values
rda_logloss = results_df[(results_df["number of sensors"]==2) & (results_df["Classifier"] == "RDA")].logloss.values
qda_logloss = results_df[(results_df["number of sensors"]==2) & (results_df["Classifier"] == "QDA")].logloss.values

st, p = friedmanchisquare(lda_logloss, rda_logloss, qda_logloss)
print("Friedman test outcome")
print("------------------------------------")
print("statistic: {:.3f}, p-value {:.3e}".format(st, p))
print('\n')

# Pair-wise comparisons
alpha = 0.05
comparison_df = pd.DataFrame(columns=["Algorithm 1", "Algorithm 2", "p-value"])
df_idx = 0
algorithm_names = ["LDA", "RDA", "QDA"]
algorithm_accuracies = [lda_logloss, rda_logloss, qda_logloss]
pvals = []
for ((name_ii, name_jj), (ii, jj)) in zip(combinations(algorithm_names, r=2), combinations(algorithm_accuracies, r=2)):
    _, p = wilcoxon(ii,jj)
    comparison_df.loc[df_idx] = [name_ii, name_jj, p]
    df_idx += 1
pvals = comparison_df["p-value"].values
num_comp = pvals.size
alpha_corrected = 1 - (1-alpha)**(1/num_comp) # Sidak comparison
comparison_df["Significant"] = comparison_df["p-value"]<alpha_corrected
print("Post-hoc pairwise comparisons:")
print("Wilcocoxon signed rank tests and Šidák correction")
print("-------------------------------------------------")
print("Alpha corrected: {:.3f}".format(alpha_corrected))
print(comparison_df.to_string(index=False))