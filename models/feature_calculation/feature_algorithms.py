# This file contains 2 functions:
#   1) Feature calculation algorithm 1: average PSD values in selected frequency bins.
#   2) Feature calculation algorithm 2: Principal Component Analysis on PSD values.


import numpy as np
# import function modules:
from models.feature_calculation import average_PSD, power_spectral_density as power, PCA_on_PSD


# Function description: calculate average power spectral density values in selected frequency bins.
# Inputs:
#   X = 3D array of signal values for multiple channels for multiple examples
#       size: (num_examples, num_channels, num_samples)
#   sample_freq = sampling frequency
#   bins = frequency bins to calculate average PSD values for
#       size: (num_bins, 2)
# Outputs:
#   PSD_bins = 3D array of average PSD values in selected frequency bins for multiple channels for multiple examples
#       size: (num_examples, num_channels, num_bins)
def average_PSD_algorithm(X, sample_freq, bins):
    # estimate power spectral density:
    Rxx, freq, PSD = power.estimate_psd(X, sample_freq)

    # calculate average PSD values in bins:
    PSD_avg = average_PSD.calc_average_PSD(PSD, bins, sample_freq)

    return PSD_avg


# Function description: performs channel-specific PCA on log-normalized power spectral density values.
# Inputs:
#   X = 3D array of signal values for multiple channels for multiple examples
#       size: (num_examples, num_channels, num_samples)
#   sample_freq = sampling frequency
#   max_freq = maximum frequency of PSD to consider
#   num_bins = number of frequency bins for average PSD calculation
#   num_pcs = number of principal components (eigenvectors) to project onto
#       validity: num_pcs <= num_freq
#   matrix_type = input to select which type of statistical matrix to calculate:
#       if matrix type == 1: autocorrelation matrices are calculated
#       if matrix type == 2: autocovariance matrices are calculated
#       else: Pearson autocovariance matrices are calculated
#   small_param = a small number to ensure that log(0) does not occur
# Outputs:
#   project_weights = 3D array of projection weights onto principal components (eigenvectors)
#       size: (num_examples, num_channels, num_pcs)
def PCA_on_PSD_algorithm(X, sample_freq, max_freq, num_bins, num_pcs, matrix_type, small_param):
    # estimate power spectral density:
    Rxx, freq, PSD = power.estimate_psd(X, sample_freq)

    # construct frequency bins for PSD average calculation:
    bin_width = max_freq / num_bins
    bins = np.zeros((num_bins, 2))
    for i in range(num_bins):
        bins[i, 0] = i * bin_width
        bins[i, 1] = (i + 1) * bin_width

    # calculate average PSD values in frequency bins:
    PSD_avg = average_PSD.calc_average_PSD(PSD, bins, sample_freq)

    # log-normalize frequency-bin-average PSD values across examples:
    PSD_norm = PCA_on_PSD.log_normalize(PSD_avg, small_param)

    # calculate selected statistical matrices of PSD values:
    if matrix_type == 1:
        stat_matrices = PCA_on_PSD.calc_correlation_matrices(PSD)
    elif matrix_type == 2:
        stat_matrices = PCA_on_PSD.calc_covariance_matrices(PSD)
    else:
        stat_matrices = PCA_on_PSD.calc_pearson_covariance_matrices(PSD)

    # calculate eigenvectors of PSD statistical matrices:
    eig_vects = PCA_on_PSD.calc_eig_vects(stat_matrices)

    # calculates projection weights of PSD values onto principal components (eigenvectors):
    project_weights = PCA_on_PSD.project_onto_pcs(PSD, eig_vects, num_pcs)

    return project_weights
