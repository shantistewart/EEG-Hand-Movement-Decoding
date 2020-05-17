# This file contains functions to create power spectral density spectrograms.


import numpy as np
from models.feature_calculation import feature_algorithms


# Function description: performs a sliding-window segmentation on raw signal values.
# Inputs:
#   X = 3D array of signal values for multiple channels for multiple examples
#       size: (num_examples, num_channels, num_samples)
#   window_size = size of sliding window, in samples
#   stride_size = size of sliding window "stride", in samples
#   num_chunks = number of chunks to split each original example into
# Outputs:
#   X_window = windowed raw signals
#       size: (num_examples, num_windows, num_channels, window_size)
def window_data(X, window_size, stride_size, num_chunks):
    # number of examples:
    num_examples = X.shape[0]
    # number of channels:
    num_channels = X.shape[1]
    # number of samples:
    num_samples = X.shape[2]

    # determine number of possible windows:
    num_windows = int(np.floor((num_samples - window_size) / stride_size) + 1)
    # make num_windows evenly divisible by num_chunks (if not already):
    num_windows = num_windows - np.mod(num_windows, num_chunks)

    X_windows = np.zeros((num_examples, num_windows, num_channels, window_size))
    for i in range(num_windows):
        # start of window index:
        start_index = i * stride_size
        # end of window index (inclusive):
        end_index = start_index + window_size - 1

        X_windows[:, i, :, :] = X[:, :, start_index:end_index+1]

    return X_windows


# Function description: creates power spectral density spectrograms.
# Inputs:
#   X_window = windowed raw signals
#       size: (num_examples, num_windows, num_channels, window_size)
#   num_chunks = number of chunks to split each original example into
#       validity: must be a factor of num_windows
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
def create_spectrogram(X_window, num_chunks, sample_freq, max_freq, num_bins, PCA, num_pcs, matrix_type, small_param):
    # number of examples:
    num_examples = X_window.shape[0]
    # number of windows:
    num_windows = X_window.shape[1]
    # number of channels:
    num_channels = X_window.shape[2]
    # window_size:
    window_size = X_window.shape[3]

    # check validity of num_chunks:
    if np.mod(num_windows, num_chunks) != 0:
        print("Invalid number of chunks.\n")
        return None

    # combine first 2 dimensions of X_window:
    #   new size: (num_examples*num_windows, num_channels, window_size)
    X_window = np.reshape(X_window, (-1, num_channels, window_size))

    # apply PCA algorithm if selected:
    if PCA == 1:
        PSD = feature_algorithms.PCA_on_PSD_algorithm(X_window, sample_freq, max_freq, num_bins, num_pcs, matrix_type, small_param)
    else:
        # construct frequency bins for PSD average calculation:
        bin_width = max_freq / num_bins
        bins = np.zeros((num_bins, 2))
        for i in range(num_bins):
            bins[i, 0] = i * bin_width
            bins[i, 1] = (i + 1) * bin_width

        # calculate average PSD values in frequency bins:
        PSD = feature_algorithms.average_PSD_algorithm(X_window, bins, sample_freq)

    # number of windows per chunk:
    num_windows_per_chunk = int(num_windows/num_chunks)
    # chunk PSD values into spectrograms:
    #   size of spectrograms: (num_examples*num_chunks, num_windows_per_chunk, num_channels, window_size)
    spectrograms = np.reshape(PSD, (num_examples*num_chunks, num_windows_per_chunk, num_channels, window_size))

    # reorder dimensions of spectrograms so that spectrograms[p, n, t, f] = spectrogram[t, f] of example p, channel n
    spectrograms = np.transpose(spectrograms, axes=(0, 2, 1, 3))

    return spectrograms
