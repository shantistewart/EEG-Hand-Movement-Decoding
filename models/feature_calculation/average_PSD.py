# This file contains 2 functions:
#   1) 1st function calculates average power spectral density values in selected frequency bins.
#   2) 2nd function plots a bar graph of average power spectral density values in selected frequency bins.


import numpy as np
import matplotlib.pyplot as plotter


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
def calc_average_PSD(PSD, bins, sample_freq):
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
        # print("Bin {0}: [{1}, {2})\n".format(i, left, right))

        # check that bin is valid:
        if left < 0 or right < 0 or left > num_samples or right > num_samples or left >= right:
            print("Invalid frequency bin.\n")
            return -1

        # calculate average:
        PSD_bins[:, :, i] = np.mean(PSD[:, :, left:right], axis=2)

    return PSD_bins


# Function description: plots a bar graph of average PSD values in selected frequency bins.
# Inputs:
#   PSD_bins = 3D array of average PSD values in selected frequency bins for multiple channels for multiple examples
#       size: (num_examples, num_channels, num_bins)
#   bins = frequency bins to calculate average PSD values for
#       size: (num_bins, 2)
#   examples = 1D array of example indices to plot:
#       size: (num_plot_examples, )
#   channels = 1D array of channel indices to plot (for all selected examples):
#       size: (num_plot_channels, )
#   channel_names = 1D array of all channel names:
#       size: (num_channels, )
# Outputs: none
def plot_average_PSD(PSD_bins, bins, examples, channels, channel_names):
    # number of frequency bins:
    num_bins = bins.shape[0]
    # number of examples to plot:
    num_plot_examples = examples.shape[0]
    # number of channels to plot (for each example):
    num_plot_channels = channels.shape[0]

    # create bin labels:
    bin_labels = []
    for i in range(num_bins):
        bin_labels.append(str(bins[i, 0]) + "-" + str(bins[i, 1]))

    # create and format subplot:
    fig, axes = plotter.subplots(num_plot_examples, num_plot_channels)
    # reshape axes to handle shape error if either dimension = 1
    axes = np.reshape(axes, (num_plot_examples, num_plot_channels))
    plotter.subplots_adjust(hspace=1)
    # locations of labels:
    bin_loc = np.arange(num_bins)
    # width of bars:
    width = 2. / num_bins

    # plot bar graph of average PSD values in selected frequency bins:
    for i in range(num_plot_examples):
        for j in range(num_plot_channels):
            axes[i, j].bar(bin_loc, PSD_bins[examples[i], channels[j]], width, label='PSD')
            axes[i, j].set_title('Average PSD of Example {0}, {1}'.format(examples[i] + 1, channel_names[channels[j]]))
            axes[i, j].set_ylabel('Average PSD')
            axes[i, j].set_xticks(bin_loc)
            axes[i, j].set_xticklabels(bin_labels)
            axes[i, j].legend()
