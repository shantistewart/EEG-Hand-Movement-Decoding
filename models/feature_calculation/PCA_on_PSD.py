# This file contains several functions to implement principal component analysis on power spectral density values.


import numpy as np


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

    # perform log-normalization across examples for each example, for each channel, for each frequency
    PSD_norm = np.log(PSD + small_param) - np.log(PSD_avg + small_param)

    return PSD_norm


# Function description: calculates channel-specific autocorrelation matrices of PSD values.
# Inputs:
#   PSD = 3D array of PSD values for multiple channels for multiple examples
#       size: (num_examples, num_channels, num_freq)
# Outputs:
#   corr_matrices = 3D array of autocorrelation matrices
#       size: (num_channels, num_freq, num_freq)
def calc_correlation_matrices(PSD):
    # number of channels:
    num_channels = PSD.shape[1]
    # number of PSD frequencies:
    num_freq = PSD.shape[2]

    # calculate autocorrelation matrices (w.r.t. examples) for each channel:
    corr_matrices = np.zeros((num_channels, num_freq, num_freq))
    for i in range(num_freq):
        for j in range(num_freq):
            # size of PSD[:, :, i] = (num_examples, num_channels)
            corr_matrices[:, i, j] = np.mean(np.multiply(PSD[:, :, i], PSD[:, :, j]), axis=0)

    return corr_matrices


# Function description: calculates channel-specific autocovariance matrices of PSD values.
# Inputs:
#   PSD = 3D array of PSD values for multiple channels for multiple examples
#       size: (num_examples, num_channels, num_freq)
# Outputs:
#   cov_matrices = 3D array of autocovariance matrices
#       size: (num_channels, num_freq, num_freq)
def calc_covariance_matrices(PSD):
    # number of channels:
    num_channels = PSD.shape[1]
    # number of PSD frequencies:
    num_freq = PSD.shape[2]

    # calculate averages across examples:
    #   size of PSD_avg = (1, num_channels, num_freq)
    PSD_avg = np.mean(PSD, axis=0, keepdims=True)

    # calculate autocovariance matrices (w.r.t. examples) for each channel:
    cov_matrices = np.zeros((num_channels, num_freq, num_freq))
    for i in range(num_freq):
        for j in range(num_freq):
            # size of PSD[:, :, i] = (num_examples, num_channels)
            cov_matrices[:, i, j] = np.mean(np.multiply(PSD[:, :, i] - PSD_avg[:, :, i], PSD[:, :, j] - PSD_avg[:, :, j]), axis=0)

    return cov_matrices


# Function description: calculates channel-specific Pearson autocovariance matrices of PSD values.
# Inputs:
#   PSD = 3D array of PSD values for multiple channels for multiple examples
#       size: (num_examples, num_channels, num_freq)
# Outputs:
#   cov_matrices = 3D array of Pearson correlation coefficient autocovariance matrices
#       size: (num_channels, num_freq, num_freq)
def calc_pearson_covariance_matrices(PSD):
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


# Function description: calculates eigenvalues/eigenvectors of PSD statistical matrices.
# Inputs:
#   matrices = 3D array of PSD statistical matrices
#       size: (num_channels, num_freq, num_freq)
# Outputs:
#   eig_vects = 3D array of normalized (unit-length) eigenvectors of PSD statistical matrices,
#       sorted in decreasing order of magnitude of corresponding eigenvalues (for each channel)
#       size: (num_channels, num_freq, num_freq)
#       eig_vects[n, :, i] = ith eigenvector of channel n
def calc_eig_vects(matrices):
    # calculate eigenvalues/eigenvectors:
    eig_vals, eig_vects = np.linalg.eigh(matrices)

    # reorder eigenvectors to be in decreasing order of magnitude of corresponding eigenvalues:
    eig_vects = np.flip(eig_vects, axis=2)

    return eig_vects


# Function description: calculates projection weights of PSD values onto principal components for all channels.
# Inputs:
#   PSD = 3D array of PSD values for multiple channels for multiple examples
#       size: (num_examples, num_channels, num_freq)
#   eig_vects = 3D array of normalized (unit-length) eigenvectors of PSD statistical matrices,
#       sorted in decreasing order of magnitude of corresponding eigenvalues (for each channel)
#       size: (num_channels, num_freq, num_freq)
#       eig_vects[n, :, i] = ith eigenvector of channel n
#   num_pcs = number of principal components (eigenvectors) to project onto
#       validity: num_pcs <= num_freq
# Outputs:
#   project_weights = 3D array of projection weights onto principal components (eigenvectors)
#       size: (num_examples, num_channels, num_pcs)
def project_onto_pcs(PSD, eig_vects, num_pcs):
    # reorder dimensions of PSD so that PSD[n, p, f] = PSD of channel n, example p, frequency f:
    #   size of reordered PSD = (num_channels, num_examples, num_freq)
    PSD = np.transpose(PSD, axes=(1, 0, 2))

    # calculate projection weights of PSD values onto num_pcs principal components for all channels:
    project_weights = np.matmul(PSD, eig_vects[:, :, :num_pcs])

    # reorder dimensions of project_weights so that project_weights[p, n, f] = projection weight of example p,
    #   channel, onto principal component f
    project_weights = np.transpose(project_weights, axes=(1, 0, 2))

    return project_weights
