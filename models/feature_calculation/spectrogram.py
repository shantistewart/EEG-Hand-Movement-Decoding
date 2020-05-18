# This file contains functions to create power spectral density spectrograms.


import numpy as np
from models.feature_calculation import feature_algorithms


# Function description: performs a sliding-window segmentation on raw signal values.
# Inputs:
#   X = 3D array of signal values for multiple channels for multiple examples
#       size: (num_examples, num_channels, num_samples)
#   window_size = size of sliding window, in samples
#   stride_size = size of sliding window "stride", in samples
# Outputs:
#   X_window = windowed raw signals
#       size: (num_examples, num_windows, num_channels, window_size)
def window_data(X, window_size, stride_size):
    # number of examples:
    num_examples = X.shape[0]
    # number of channels:
    num_channels = X.shape[1]
    # number of samples:
    num_samples = X.shape[2]

    # determine number of possible windows:
    num_windows = int(np.floor((num_samples - window_size) / stride_size) + 1)

    X_window = np.zeros((num_examples, num_windows, num_channels, window_size))
    for i in range(num_windows):
        # start of window index:
        start_index = i * stride_size
        # end of window index (inclusive):
        end_index = start_index + window_size - 1

        X_window[:, i, :, :] = X[:, :, start_index:end_index+1]

    return X_window


# Function description: creates power spectral density spectrograms.
# Inputs:
#   X_window = windowed raw signals
#       size: (num_examples, num_windows, num_channels, window_size)
#   sample_freq = sampling frequency
#   max_freq = maximum frequency of PSD to consider
#   num_bins = number of frequency bins for average PSD calculation
#   PCA = parameter to select whether to apply PCA algorithm
#       if PCA == 1: PCA algorithm is applied
#       else: PCA algorithm is not applied
#   num_pcs = number of principal components (eigenvectors) to project onto
#       validity: num_pcs <= num_freq
#   matrix_type = parameter to select which type of statistical matrix to calculate:
#       if matrix type == 1: autocorrelation matrices are calculated
#       if matrix type == 2: autocovariance matrices are calculated
#       else: Pearson autocovariance matrices are calculated
#   small_param = a small number to ensure that log(0) does not occur for log-normalization
# Outputs:
#   spectrograms = PSD spectrograms
#       size: (num_examples, num_channels, num_windows, num_bins)
def create_spectrogram(X_window, sample_freq, max_freq, num_bins, PCA, num_pcs, matrix_type, small_param):
    # number of examples:
    num_examples = X_window.shape[0]
    # number of windows:
    num_windows = X_window.shape[1]

    # combine first 2 dimensions of X_window:
    #   new size: (num_examples*num_windows, num_channels, window_size)
    X_window = np.reshape(X_window, (-1, X_window.shape[2], X_window.shape[3]))

    # apply PCA algorithm if selected:
    if PCA == 1:
        PSD = feature_algorithms.PCA_on_PSD_algorithm(X_window, sample_freq, max_freq, num_bins, num_pcs, matrix_type,
                                                      small_param)
    else:
        # construct frequency bins for PSD average calculation:
        bin_width = max_freq / num_bins
        bins = np.zeros((num_bins, 2))
        for i in range(num_bins):
            bins[i, 0] = i * bin_width
            bins[i, 1] = (i + 1) * bin_width

        # calculate average PSD values in frequency bins:
        PSD = feature_algorithms.average_PSD_algorithm(X_window, bins, sample_freq)

    # undo combining of first 2 dimensions of X_window:
    spectrograms = np.reshape(PSD, (num_examples, num_windows, PSD.shape[1], PSD.shape[2]))
    # reorder dimensions of spectrograms so that spectrograms[p, n, t, f] = spectrogram[t, f] of example p, channel n
    spectrograms = np.transpose(spectrograms, axes=(0, 2, 1, 3))

    return spectrograms
