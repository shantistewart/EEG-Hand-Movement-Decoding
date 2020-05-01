# This file contains several functions to implement principal component analysis on power spectral density features.


import numpy as np
import numpy.linalg as lin_alg


# Function description: performs log-normalization of PSD values across all examples.
# Inputs:
#   PSD = 3D array of (non-negative) PSD values for multiple channels for multiple examples
#       size: (num_examples, num_channels, num_freq)
#   small_param = a small number to ensure that log(0) does not occur
# Outputs:
#   PSD_norm = 3D array of log-normalized (non-negative) PSD values for multiple channels for multiple examples
#       size: (num_examples, num_channels, num_freq)
def log_normalize(PSD, small_param):
    # calculate averages across examples:
    PSD_avg = np.mean(PSD, axis=0, keepdims=True)
    print("Example-average PSD values:\nSize: ", end="")
    print(PSD_avg.shape)
    print(PSD_avg)
    print("")

    # perform log-normalization across examples for each example, for each channel, for each frequency
    PSD_norm = np.log(PSD + small_param) - np.log(PSD_avg + small_param)

    return PSD_norm


# Function description: calculates unnormalized channel-specific autocovariance matrices.
# Inputs:
#   PSD = 3D array of (non-negative) PSD values for multiple channels for multiple examples
#       size: (num_examples, num_channels, num_freq)
# Outputs:
#   cov_matrices = 3D array of unnormalized autocovariance matrices
#       size: (num_channels, num_freq, num_freq)
def unnorm_covariance(PSD):
    # number of channels:
    num_channels = PSD.shape[1]
    # number of PSD frequencies:
    num_freq = PSD.shape[2]

    # calculate averages across examples:
    PSD_avg = np.mean(PSD, axis=0, keepdims=True)
    print("Example-average PSD values:\nSize: ", end="")
    print(PSD_avg.shape)
    print(PSD_avg)
    print("")

    # calculate unnormalized autocovariance matrices (w.r.t. examples) for each channel:
    cov_matrices = np.zeros((num_channels, num_freq, num_freq))
    for i in range(num_freq):
        for j in range(num_freq):
            cov_matrices[:, i, j] = np.sum(np.multiply(PSD[:, :, i], PSD[:, :, j]), axis=0)
    print("Channel-specific autocovariance matrices:\nSize: ", end="")
    print(cov_matrices.shape)
    print(cov_matrices)
