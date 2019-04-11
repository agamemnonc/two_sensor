import pandas as pd
from scipy.stats import wilcoxon
import warnings

from utils import get_df_early_late

warnings.filterwarnings("ignore")

df = pd.read_csv('data/raw_emg_power_trial.csv')
df_early_late = get_df_early_late(df)

# Get early and late EMG variance as numpy arrays
for electrodes in ['Used', 'Not used']:
    early_var = df_early_late[(df_early_late["Electrodes"]==electrodes) &
                             (df_early_late["Phase"]=='Early')]['Average EMG variance']
    late_var = df_early_late[(df_early_late["Electrodes"]==electrodes) &
                             (df_early_late["Phase"]=='Late')]['Average EMG variance']
    
    print("{} electrodes".format(electrodes))
    print("Wilcoxon signed-rank test (early vs. late trials)")
    print("-------------------------------------------------")
    print("p={:.3f}, n={}".format(wilcoxon(early_var, late_var)[1],
          early_var.size))
    print("\n")