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
    PSD_bins = np.zeros((num_examples, num_channels, num_bins))
    for i in range(num_bins):
        # left sample index (round down):
        left = int(np.floor((bins[i, 0] / max_freq) * num_samples))
        # right sample index (round up):
        right = int(np.ceil((bins[i, 1] / max_freq) * num_samples))
        print("Bin {0}: [{1}, {2})\n".format(i, left, right))

        # check that bin is valid:
        if left < 0 or right < 0 or left > num_samples or right > num_samples or left >= right:
            print("Invalid frequency bin.\n")
            return -1

        # calculate average:
        PSD_bins[:, :, i] = np.mean(PSD[:, :, left:right], axis=2)

    return PSD_bins
