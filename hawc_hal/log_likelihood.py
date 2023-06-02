from numba import jit
import numpy as np


# This function has two signatures in numba because if there are no sources in the likelihood model,
# then expected_model_counts is 0.0
@jit(["float64(float64[:], float64[:], float64[:])", "float64(float64[:], float64[:], float64)"],
     nopython=True, parallel=False)
def log_likelihood(observed_counts, expected_bkg_counts, expected_model_counts):  # pragma: no cover
    """
    Poisson log-likelihood minus log factorial minus bias. The bias migth be needed to keep the numerical value
    of the likelihood small enough so that there aren't numerical problems when computing differences between two
    likelihood values.

    :param observed_counts:
    :param expected_bkg_counts:
    :param expected_model_counts:
    :param bias:
    :return:
    """

    predicted_counts = expected_bkg_counts + expected_model_counts

    # Remember: because of how the DataAnalysisBin in map_tree.py initializes the maps,
    # observed_counts > 0 everywhere

    log_likes = observed_counts * np.log(predicted_counts) - predicted_counts

    return np.sum(log_likes)