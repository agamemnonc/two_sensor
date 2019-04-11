import pandas as pd
from scipy.stats import wilcoxon
import warnings

from utils import get_early_late_times, get_df_early_late

warnings.filterwarnings("ignore")

# Load data
df = pd.read_csv('data/experimental_results.csv')
df_average_time_block_type  = get_df_early_late(df) # Early vs. late
# Get early_times and late_times as numpy arrays
early_times, late_times = get_early_late_times(df_average_time_block_type)

print("Wilcoxon signed-rank test (early vs. late trials)")
print("-------------------------------------------------")
print("p={:.3f}, n={}".format(wilcoxon(early_times, late_times)[1],
      early_times.size))