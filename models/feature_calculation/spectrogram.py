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
#       size: (num_chunks * num_windows * num_examples, num_channels, num_samples)
#   num_windows = number of windows per chunk
def window_data(X, window_size, stride_size, num_chunks):

    return X


# Function description: creates power spectral density spectrograms.
# Inputs:
#   X_window = windowed raw signals
#       size: (num_chunks * num_windows * num_examples, num_channels, num_samples)
#   num_windows = number of windows per chunk
#   num_chunks = number of chunks to split each original example into
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
def create_spectrogram(X, num_windows, sample_freq, max_freq, num_bins, PCA, num_pcs, matrix_type, small_param):

    return X
