# This file contains 3 functions:
#   1) Feature calculation algorithm 1: frequency bin-average PSD.
#   2) Feature calculation algorithm 2: PCA on PSD.
#   3) Feature calculation algorithm 3: PSD spectrograms.


import numpy as np
from models.feature_calculation import power_spectral_density as power, average_PSD, PCA_on_PSD, spectrogram
from models.classifiers import example_generation


# Function description: calculate average power spectral density values in selected frequency bins.
# Inputs:
#   X = 3D array of signal values for multiple channels for multiple examples
#       size: (num_examples, num_channels, num_samples)
#   bins = frequency bins to calculate average PSD values for
#       size: (num_bins, 2)
#   sample_freq = sampling frequency
# Outputs:
#   PSD_avg = 3D array of average PSD values in selected frequency bins for multiple channels for multiple examples
#       size: (num_examples, num_channels, num_bins)
def average_PSD_algorithm(X, bins, sample_freq):
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
#       validity: num_pcs <= num_bins
#   matrix_type = input to select which type of statistical matrix to calculate:
#       if matrix type == 1: autocorrelation matrices are calculated
#       if matrix type == 2: autocovariance matrices are calculated
#       else: Pearson autocovariance matrices are calculated
#   small_param = a small number to ensure that log(0) does not occur for log-normalization
# Outputs:
#   project_weights = 3D array of projection weights onto principal components (eigenvectors)
#       size: (num_examples, num_channels, num_pcs)
def PCA_on_PSD_algorithm(X, sample_freq, max_freq, num_bins, num_pcs=None, matrix_type=2, small_param=0.0001):
    # construct frequency bins for PSD average calculation:
    bin_width = max_freq / num_bins
    bins = np.zeros((num_bins, 2))
    for i in range(num_bins):
        bins[i, 0] = i * bin_width
        bins[i, 1] = (i + 1) * bin_width

    # calculate average PSD values in frequency bins:
    PSD_avg = average_PSD_algorithm(X, bins, sample_freq)

    # log-normalize frequency-bin-average PSD values across examples:
    # PSD_avg = PCA_on_PSD.log_normalize(PSD_avg, small_param)

    # calculate selected statistical matrices of PSD values:
    if matrix_type == 1:
        stat_matrices = PCA_on_PSD.calc_correlation_matrices(PSD_avg)
    elif matrix_type == 2:
        stat_matrices = PCA_on_PSD.calc_covariance_matrices(PSD_avg)
    else:
        stat_matrices = PCA_on_PSD.calc_pearson_covariance_matrices(PSD_avg)

    # calculate eigenvectors of PSD statistical matrices:
    eig_vects = PCA_on_PSD.calc_eig_vects(stat_matrices)

    # default num_pcs to num_bins:
    if num_pcs is None:
        num_pcs = num_bins
    # calculates projection weights of PSD values onto principal components (eigenvectors):
    project_weights = PCA_on_PSD.project_onto_pcs(PSD_avg, eig_vects, num_pcs)

    return project_weights


# Function description: creates power spectral density spectrograms, with an option to apply the PCA algorithm.
# Inputs:
#   X = 3D array of signal values for multiple channels for multiple examples
#       size: (num_examples, num_channels, num_samples)
#   window_size = size of sliding window to calculate PSD, in seconds
#   stride_size = size of "stride" of sliding window to calculate PSD, in seconds
#   sample_freq = sampling frequency
#   max_freq = maximum frequency of PSD to consider
#   num_bins = number of frequency bins for average PSD calculation
#   PCA = parameter to select whether to apply PCA algorithm
#       if PCA == 1: PCA algorithm is applied
#       else: PCA algorithm is not applied
#   num_pcs = number of principal components (eigenvectors) to project onto
#       validity: num_pcs <= num_bins
#   matrix_type = parameter to select which type of statistical matrix to calculate:
#       if matrix type == 1: autocorrelation matrices are calculated
#       if matrix type == 2: autocovariance matrices are calculated
#       else: Pearson autocovariance matrices are calculated
#   small_param = a small number to ensure that log(0) does not occur for log-normalization
# Outputs:
#   spectrograms = PSD spectrograms
#       size: (num_examples, num_channels, num_windows, num_bins)
def spectrogram_algorithm(X, window_size, stride_size, sample_freq, max_freq, num_bins, PCA=0, num_pcs=None,
                          matrix_type=2, small_param=0.0001):
    # convert window and stride sizes from seconds to samples:
    window_size = int(np.floor(sample_freq * window_size))
    stride_size = int(np.floor(sample_freq * stride_size))

    # window data for PSD estimation
    X_window = example_generation.window_data(X, window_size, stride_size)

    # create spectrograms, optionally with the PCA algorithm:
    spectrograms = spectrogram.create_spectrogram(X_window, sample_freq, max_freq, num_bins, PCA=PCA, num_pcs=num_pcs,
                                                  matrix_type=matrix_type, small_param=small_param)

    return spectrograms
