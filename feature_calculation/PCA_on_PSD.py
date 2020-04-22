# This file contains several functions to implement principal component analysis on power spectral density features.


import numpy as np
import numpy.linalg as lin_alg


# Function description: performs log-normalization of PSD values across all examples.
# Inputs:
#   PSD = 3D array of (non-negative) PSD values for multiple channels for multiple examples
#       size: (num_examples, num_channels, num_samples)
#   small_param = a small number to ensure that log(0) does not occur
# Outputs:
#   PSD_norm = 3D array of log-normalized PSD values for multiple channels for multiple examples
#       size: (num_examples, num_channels, num_samples)
# def log_normalize(PSD, small_param):
    # calculate averages across examples:
    # PSD_avg = np.mean

    # return
