# This file contains a function that calculates average power spectral density values in selected frequency bins.


import numpy as np


# Function description: calculate average power spectral density values in selected frequency bins.
# Inputs:
#   PSD = 3D array of (non-negative) PSD values for multiple channels for multiple examples
#       size: (num_examples, num_channels, num_samples)
#   bins = frequency bins to calculate average PSD values for
#       size: (num_bins, 2)
#   sample_freq = sampling frequency of original signal
# Outputs:
#   PSD_bins = 3D array of average PSD values in selected frequency bins for multiple channels for multiple examples
#       size: (num_examples, num_channels, num_bins)
def average_PSD(PSD, bins, sample_freq):
    # number of examples:
    num_examples = PSD.shape[0]
    # number of channels:
    num_channels = PSD.shape[1]
    # number of samples:
    num_samples = PSD.shape[2]
    # number of frequency bins:
    num_bins = bins.shape[0]
    # max frequency (Nyquist frequency):
    max_freq = .5 * sample_freq

    # calculate average PSD values in frequency bins:
    PSD_bins = np.zeros(num_bins)



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

    # return average power spectral density values in frequency bins:
    return PSD_bins
