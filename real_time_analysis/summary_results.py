import numpy as np
import pandas as pd
import warnings

from utils import get_early_late_times, get_df_early_late, get_df_mean_rates

warnings.filterwarnings("ignore")

# Load data
df = pd.read_csv('data/experimental_results.csv')
df_mean = get_df_mean_rates(df) # Results with mean completion rates
df_average_time_block_type  = get_df_early_late(df) # Early vs. late
# Get early_times and late_times as numpy arrays
early_times, late_times = get_early_late_times(df_average_time_block_type)

# Display median completion rates
print("\n")
print("Median completion rates")
print("------------------------------")
print("Able-bodied participants: {:.1f}".format(
        df_mean[df_mean['Participant']=='Able-bodied'][
                'Mean completion rate'].median()))
print("Amputee participants: {:.1f}".format(
        df_mean[df_mean['Participant']=='Amputee'][
                'Mean completion rate'].median()))

# Display median completion times
print("\n")
print("Median completion times")
print("------------------------------")
print("Able-bodied participants: {:.2f}".format(
        df[df['Participant']=='Able-bodied'][
                'Trial_time'].median()))
print("Amputee participants: {:.2f}".format(
        df[df['Participant']=='Amputee'][
                'Trial_time'].median()))

# Display difference between early and late times
print("\n")
print("Early vs. late trials")
print("------------------------------")
print("Difference in median completion times: {:.2f}".format(
        np.median(early_times)-np.median(late_times)))