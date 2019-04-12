import numpy as np
import pandas as pd

# Subject numbers convention
# Able-bodied: 1-12
# Amputee: 20-21
SUBJECTS = np.concatenate((np.arange(1,13), np.array([20,21]))).astype(
        np.int32)

def get_df_mean_rates(df):
    """Creates a new data frame with mean completion rates for each
    participant.
    
    Parameters
    ----------
    df : DataFrame
        Data frame with results for each participant. Must have a column
        ```Trial_success```.
    
    Returns
    -------
    df_mean : DataFrame
        Data frame including mean completion rates.
    """
    df_mean = pd.DataFrame(columns=['Subject number', "Participant",
                                    'Mean completion rate'])
    # Able-bodied
    i = 0
    for subject in np.unique(df['Subject number']):
        df_sub = df[df["Subject number"]==subject]
        mn = df_sub['Trial_success'].mean()
        df_mean.loc[i] = [int(subject), df_sub.Participant.unique()[0], 100*mn]
        i += 1
    
    return df_mean

def get_df_early_late(df, early_limit=3, late_limit=7):
    """Creates a new data frame for storing results where trials are
    categorised as either ```early``` or ```late```.
    
    Parameters
    ----------
    df : DataFrame
        Data frame with results for each participant. 
    
    early_limit : int
        Limit for a trial to be considered as ```early``` (not inclusive).
    
    late_limit : int
        Limit for a trial to be considered as ```late``` (not inclusive).
    
    Returns
    -------
    df_avg_time_block : DataFrame
        Data frame with average completion times for early and late trials.
    """
    average_function = np.median
    df_avg_time_block = pd.DataFrame(columns=['Subject number',
                                              'Participant',
                                              'Block type',
                                              'Average completion time'])
    i = 0
    for subject in SUBJECTS:
        if subject == 20 or subject == 21:
            participant = "Amputee"
        else:
            participant = "Able-bodied"
        avg_early_time = average_function(
            df[(df["Subject number"]==subject) & 
               (df["Trial_success"]==1) &
              (df["Trial"]<early_limit)]["Trial_time"])
        avg_late_time = average_function(
            df[(df["Subject number"]==subject) & 
               (df["Trial_success"]==1) &
              (df["Trial"]>late_limit)]["Trial_time"])
        df_avg_time_block.loc[i] = [subject, participant, 'early',
                             avg_early_time]
        df_avg_time_block.loc[i+1] = [subject, participant, 'late',
                             avg_late_time]
        i = i + 2
    
    return df_avg_time_block
    
def get_early_late_times(df_average):
    """Returns early and late trial completion times for statistical
    comparisons.
    
    Parameters
    ----------
    df_average : DataFrame
        Data frame with results for each participant.
    
    Returns
    -------
    early_times : array
        Array with early times.
    late_times : array
        Array with late times.
    """
    early_times = df_average[df_average[
            "Block type"]=='early']['Average completion time'].values
    late_times = df_average[df_average[
            "Block type"]=='late']['Average completion time'].values
    early_none = np.where(np.isnan(early_times))
    late_none = np.where(np.isnan(late_times))
    
    early_times = np.delete(early_times, np.concatenate((early_none[0],
                                                         late_none[0])))
    late_times = np.delete(late_times, np.concatenate((early_none[0],
                                                       late_none[0])))
    
    return (early_times, late_times)