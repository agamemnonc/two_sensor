import numpy as np
import pandas as pd

def load_results(path):
    """Loads results file.
    
    Parameters
    ----------
    path : str
        Path were results file is stored.
    
    Returns
    -------
    results_df : DataFrame
        Pandas DataFrame with results.
    """
    results_df = pd.read_csv(path)
    return results_df

def load_confusion_matrices(path, classifier='RDA', n_sensors=2):
    """Loads offline confusion matrices and computes averages across able-
    bodied and amputee populations for a specified classifier.
    
    Parameters
    ----------
    path : str
        Path were confusion matrix results are stored.
    classifier : str (default: 'RDA')
        Classifier of interest. One of ['LDA', 'RDA', 'QDA'].
    n_sensors : int (default: 2)
        Number of sensors for comparison.
    
    Returns
    -------
    cm_ab_mean : array, shape=(n_classes,n_classes)
        Confusion matrix average for able-bodied population.
    cm_amp_mean : array, shape=(n_classes,n_classes)
        Confusion matrix average for amputee population.
    labels : list of str
        Class labels.
    """
    CLASSIFIERS = ["LDA", "RDA", "QDA"]
    LABELS = ['rest', 'power', 'lateral', 'tripod', 'pointer', 'open']
    N_AB = 12 # number of able-bodied participants
    N_AMP = 2 # number of amptuee participants
    
    confusion_matrices = np.load(path)
    cm_ab = confusion_matrices[0:N_AB, n_sensors-1,
                               CLASSIFIERS.index(classifier), :, :]
    cm_amp = confusion_matrices[N_AB:N_AB + N_AMP, n_sensors-1,
                                CLASSIFIERS.index(classifier), :, :]
    cm_ab_mean = np.mean(cm_ab, axis=0)
    cm_amp_mean = np.mean(cm_amp, axis=0)
    
    return (cm_ab_mean, cm_amp_mean, LABELS)