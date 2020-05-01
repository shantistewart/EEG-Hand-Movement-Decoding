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
    """
    print("Example-average PSD values:\nSize: ", end="")
    print(PSD_avg.shape)
    print(PSD_avg)
    print("")
    """

    # perform log-normalization across examples for each example, for each channel, for each frequency
    PSD_norm = np.log(PSD + small_param) - np.log(PSD_avg + small_param)

    return PSD_norm


# Function description: calculates unnormalized channel-specific autocorrelation matrices.
# Inputs:
#   PSD = 3D array of (non-negative) PSD values for multiple channels for multiple examples
#       size: (num_examples, num_channels, num_freq)
# Outputs:
#   corr_matrices = 3D array of unnormalized autocorrelation matrices
#       size: (num_channels, num_freq, num_freq)
def unnorm_correlation(PSD):
    # number of channels:
    num_channels = PSD.shape[1]
    # number of PSD frequencies:
    num_freq = PSD.shape[2]

    # calculate unnormalized autocorrelation matrices (w.r.t. examples) for each channel:
    corr_matrices = np.zeros((num_channels, num_freq, num_freq))
    for i in range(num_freq):
        for j in range(num_freq):
            # size of PSD[:, :, i] = (num_examples, num_channels)
            corr_matrices[:, i, j] = np.sum(np.multiply(PSD[:, :, i], PSD[:, :, j]), axis=0)

    return corr_matrices


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
    #   size of PSD_avg = (1, num_channels, num_freq)
    PSD_avg = np.mean(PSD, axis=0, keepdims=True)
    # """
    print("Example-average PSD values:\nSize: ", end="")
    print(PSD_avg.shape)
    print(PSD_avg)
    print("")
    # """

    # calculate unnormalized autocovariance matrices (w.r.t. examples) for each channel:
    cov_matrices = np.zeros((num_channels, num_freq, num_freq))
    for i in range(num_freq):
        for j in range(num_freq):
            # size of PSD[:, :, i] = (num_examples, num_channels)
            cov_matrices[:, i, j] = np.sum(np.multiply(PSD[:, :, i]-PSD_avg[:, :, i], PSD[:, :, j]-PSD_avg[:, :, j]), axis=0)

    return cov_matrices


# Function description: calculates Pearson correlation coefficient channel-specific autocovariance matrices.
# Inputs:
#   PSD = 3D array of (non-negative) PSD values for multiple channels for multiple examples
#       size: (num_examples, num_channels, num_freq)
# Outputs:
#   cov_matrices = 3D array of Pearson correlation coefficient autocovariance matrices
#       size: (num_channels, num_freq, num_freq)
def pearson_covariance(PSD):
    # number of channels:
    num_channels = PSD.shape[1]
    # number of PSD frequencies:
    num_freq = PSD.shape[2]

    # calculate Pearson autocovariance matrices (w.r.t. examples) for each channel:
    cov_matrices = np.zeros((num_channels, num_freq, num_freq))
    for n in range(num_channels):
        # size of PSD[:, n, :] = (num_examples, num_freq)
        cov_matrices[n] = np.corrcoef(PSD[:, n, :], rowvar=False)

    return cov_matrices
