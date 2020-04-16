# This file contains 2 functions:
#   1) First function estimates power spectral density.
#   2) Second function calculates average values of power spectral density over specified frequency bins.


import numpy as np


# Function description: estimates power spectral density.
# Inputs:
#   X = 3D array of signal values for multiple channels for multiple examples
#       size: (num_examples, num_channels, num_samples)
#   sample_freq = sampling frequency
# Outputs:
#   Rxx_pos = 3D array of (non-negative) autocorrelation functions for multiple channels for multiple examples
#       size: (num_examples, num_channels, num_samples)
#   freq_pos = non-negative frequencies of PSD (same for all examples for all channels)
#       size: (1, 1, num_samples)
#   PSD_pos = 3D array of (non-negative) power spectral density values for multiple channels for multiple examples
#       size: (num_examples, num_channels, num_samples)
def estimate_psd(X, sample_freq):
    # number of examples:
    num_examples = X.shape[0]
    # number of channels:
    num_channels = X.shape[1]
    # number of samples:
    num_samples = X.shape[2]
    # sampling period:
    sample_period = 1 / sample_freq

    # estimate non-normalized 2nd-order moments (over all samples) for each example for each channel:
    second_moment = np.sum(np.multiply(X, X), axis=2, keepdims=True)
    # print("Non-normalized seconds moments:")
    # print(second_moment)
    # print("")

    # estimate non-negative part of autocorrelation function for each example for each channel:
    Rxx_pos = np.zeros((num_examples, num_channels, num_samples))
    for k in range(0, num_samples):
        # number of samples of overlap between X[:, :, n] and X[:, :, n-k]:
        overlap = num_samples - k
        # X[:, :, n] with X[:, :, 0...k-1] elements removed (equivalently zero-padded):
        X_n = X[:, :, k:]
        # X[:, :, n-k] with X[:, :, num_samples-k...num_samples-1] elements removed (equivalently zero-padded):
        X_shift = X[:, :, :overlap]
        # calculate vector inner product (over all samples) for each example for each channel:
        Rxx_pos[:, :, k] = np.sum(np.multiply(X_n, X_shift), axis=2)
    # normalize non-negative part of autocorrelation functions:
    Rxx_pos = np.divide(Rxx_pos, second_moment)

    # calculate power spectral density =  Fourier transform of autocorrelation function (for each example for each
    #   channel), using Hermitian FFT (since autocorrelation function is symmetric):
    PSD = np.fft.hfft(Rxx_pos, axis=2)
    # extract non-negative part of PSD:
    PSD_pos = PSD[:, :, :num_samples]
    # normalize PSD_pos:
    PSD_pos = PSD_pos / num_samples
    # number of frequencies (positive and negative) of PSD:
    num_freq = PSD.shape[2]
    # normalize PSD_pos:
    # PSD_pos = PSD_pos / num_freq

    # frequencies of PSD (same for all examples for all channels):
    freq = np.fft.fftfreq(num_freq, d=sample_period)
    # extract non-negative frequencies (and negative Nyquist frequency):
    freq_pos = freq[:num_samples]
    # change negative Nyquist frequency to positive Nyquist frequency:
    freq_pos[num_samples-1] = -1*freq_pos[num_samples-1]
    # reshape freq_pos in order to be broadcastable with PSD_pos:
    freq_pos = np.reshape(freq_pos,(1, 1, freq_pos.shape[0]))

    return Rxx_pos, freq_pos, PSD_pos
