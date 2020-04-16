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

    # frequencies of PSD (same for all examples for all channels):
    freq = np.fft.fftfreq(num_freq, d=sample_period)
    # extract non-negative frequencies (and negative Nyquist frequency):
    freq_pos = freq[:num_samples]
    # change negative Nyquist frequency to positive Nyquist frequency:
    freq_pos[num_samples-1] = -1*freq_pos[num_samples-1]
    # reshape freq_pos in order to be broadcastable with PSD_pos:
    freq_pos = np.reshape(freq_pos, (1, 1, freq_pos.shape[0]))

    return Rxx_pos, freq_pos, PSD_pos


# Function description: calculate average PSD values in specified frequency bins.
# Inputs:
#   PSD = 3D array of (non-negative) power spectral density values for multiple channels for multiple examples
#       size: (num_examples, num_channels, num_samples)
#   bins = frequency bins to calculate average PSD values for
#       size: (num_bins, 2)
#   sample_freq = sampling frequency of original signal
# Outputs:
#   PSD_bins = 3D array of average PSD values in selected frequency bins for multiple channels for multiple examples
#       size: (num_examples, num_channels, num_bins)
def PSD_average(PSD, bins, sample_freq):
    # number of examples:
    num_examples = PSD.shape[0]
    # number of channels:
    num_channels = PSD.shape[1]
    # number of samples:
    num_samples = PSD.shape[2]
    # number of frequency bins:
    num_bins = bins.shape[0]



# Function description: calculate average PSD values in specified frequency bins.
# Inputs:
#   PSD_mag_pos = 1D array of magnitude response of PSD
#       size: [pos_num_samples]
#   bins = frequency bins of interest
#       size: [num_bins, 2]
#   bin_labels = frequency bin labels
#       size: [num_bins]
#   sample_freq = sampling frequency of original signal
#   disp = bool to select whether to display bar graph
# Outputs:
#   PSD_bins = 1D array of average PSD values in frequency bins
#       size: [num_bins]
def average_bin(PSD_mag_pos, bins, bin_labels, sample_freq, disp):
    # number of PSD samples:
    num_samples = len(PSD_mag_pos)
    # max sampling frequency (Nyquist rate):
    max_freq = .5 * sample_freq
    # number of frequency bins:
    num_bins = len(bins)

    # calculate average power spectral density values in frequency bins:
    PSD_bins = np.zeros(num_bins)
    for i in range(num_bins):
        # left index:
        left = int(np.floor(num_samples * (bins[i, 0] / max_freq)))
        # right index:
        right = int(np.floor(num_samples * (bins[i, 1] / max_freq)))
        # number of samples in bin:
        bin_width = right - left
        PSD_bins[i] = np.sum(PSD_mag_pos[left:right]) / bin_width

    # bar graph to display PSD average bin values:
    if disp:
        # locations of labels:
        bin_loc = np.arange(num_bins)
        # width of bars:
        width = 2. / num_bins
        # create figure:
        fig, ax = plotter.subplots()
        # create bar chart:
        ax.bar(bin_loc, PSD_bins, width, label='Average PSD')
        # add title, labels, and legend:
        ax.set_title('Average PSD in Frequency Bins')
        ax.set_ylabel('Average PSD')
        ax.set_xticks(bin_loc)
        ax.set_xticklabels(bin_labels)
        ax.legend()

    # return average power spectral density values in frequency bins:
    return PSD_bins
