# This file contains the definition of the class RawPSD.


import numpy as np
import matplotlib.pyplot as plotter


# Class description: stores raw signals, autocorrelation functions, and power spectral densities for multiple channels
#   for multiple examples, and has plotting functions for all 3.
# Instance variables:
#   num_examples = number of examples
#   num_channels = number of channels
#   num_samples = number of samples
#   sample_freq = sampling frequency
#   sample_period = sampling period
#   X = 3D array of raw signal values for multiple channels for multiple examples
#       size: (num_examples, num_channels, num_samples)
#   Rxx = 3D array of (non-negative) autocorrelation functions for multiple channels for multiple examples
#       size: (num_examples, num_channels, num_samples)
#   freq = non-negative frequencies of PSD (same for all examples for all channels)
#       size: (1, 1, num_samples)
#   PSD = 3D array of (non-negative) power spectral density values for multiple channels for multiple examples
#       size: (num_examples, num_channels, num_samples)
# Inputs to constructor: all instance variables except sample_period
# Methods:
#   plot_raw_signal(): plots raw signals of selected channels of selected examples.
#   plot_autocorr(): plots autocorrelation functions of selected channels of selected examples.
class RawPSD:
    """Class of raw power spectral density values, autocorrelation functions, and raw signals."""

    # Constructor:
    def __init__(self, M, N, L, sampling_frequency, raw_signal, autocorrelation, frequencies, power_spectral_density):
        self.num_examples = M
        self.num_channels = N
        self.num_samples = L
        self.sample_freq = sampling_frequency
        self.sample_period = 1 / sampling_frequency
        self.X = raw_signal
        self.Rxx = autocorrelation
        self.freq = frequencies
        self.PSD = power_spectral_density

    # Methods:

    # Function description: plots raw signals of selected channels of selected examples.
    # Inputs:
    #   examples = 1D array of example indices to plot:
    #       size: (num_plot_examples, )
    #   channels = 1D array of channel indices to plot (for all selected examples):
    #       size: (num_plot_channels, )
    #   channel_names = 1D array of all channel names:
    #       size: (num_channels, )
    # Outputs: none
    def plot_raw_signal(self, examples, channels, channel_names):
        # number of examples to plot:
        num_plot_examples = examples.shape[0]
        # number of channels to plot (for each example):
        num_plot_channels = channels.shape[0]

        # sample points of raw signals:
        samples = np.arange(0, self.num_samples)
        # actual time points of raw signals:
        time = self.sample_period * samples

        # create and format subplot:
        fig, axes = plotter.subplots(num_plot_channels, num_plot_examples)
        # reshape axes to handle shape error if either dimension = 1
        axes = np.reshape(axes, (num_plot_channels, num_plot_examples))
        plotter.subplots_adjust(hspace=1)

        # plot raw signals of selected examples and channels:
        for i in range(num_plot_examples):
            for j in range(num_plot_channels):
                axes[j, i].set_title('x[n] of Example {0}, {1}'.format(examples[i] + 1, channel_names[channels[j]]))
                axes[j, i].plot(time, self.X[examples[i], channels[j]])
                axes[j, i].set_xlabel('Time (s)')
                axes[j, i].set_ylabel('x[n]')

    # Function description: plots autocorrelation functions of selected channels of selected examples.
    # Inputs:
    #   examples = 1D array of example indices to plot:
    #       size: (num_plot_examples, )
    #   channels = 1D array of channel indices to plot (for all selected examples):
    #       size: (num_plot_channels, )
    #   channel_names = 1D array of all channel names:
    #       size: (num_channels, )
    # Outputs: none
    def plot_autocorr(self, examples, channels, channel_names):
        # number of examples to plot:
        num_plot_examples = examples.shape[0]
        # number of channels to plot (for each example):
        num_plot_channels = channels.shape[0]

        # sample points of raw signals:
        samples = np.arange(0, self.num_samples)

        # create and format subplot:
        fig, axes = plotter.subplots(num_plot_channels, num_plot_examples)
        # reshape axes to handle shape error if either dimension = 1
        axes = np.reshape(axes, (num_plot_channels, num_plot_examples))
        plotter.subplots_adjust(hspace=1)

        # plot autocorrelation functions of selected examples and channels:
        for i in range(num_plot_examples):
            for j in range(num_plot_channels):
                axes[j, i].set_title('Rxx[k] of Example {0}, {1}'.format(examples[i] + 1, channel_names[channels[j]]))
                axes[j, i].plot(samples, self.Rxx[examples[i], channels[j]])
                axes[j, i].set_xlabel('Samples')
                axes[j, i].set_ylabel('Rxx[k]')

# Function description: plots power spectral densities of selected channels of selected examples.
# Inputs:
#   examples = 1D array of example indices to plot:
#       size: (num_plot_examples, )
#   channels = 1D array of channel indices to plot (for all selected examples):
#       size: (num_plot_channels, )
#   channel_names = 1D array of all channel names:
#       size: (num_channels, )
# Outputs: none
    def plot_PSD(self, examples, channels, channel_names):
        # number of examples to plot:
        num_plot_examples = examples.shape[0]
        # number of channels to plot (for each example):
        num_plot_channels = channels.shape[0]

        # create and format subplot:
        fig, axes = plotter.subplots(num_plot_channels, num_plot_examples)
        # reshape axes to handle shape error if either dimension = 1
        axes = np.reshape(axes, (num_plot_channels, num_plot_examples))
        plotter.subplots_adjust(hspace=1)

        # plot power spectral densities of selected examples and channels:
        for i in range(num_plot_examples):
            for j in range(num_plot_channels):
                axes[j, i].set_title('PSD of Example {0}, {1}'.format(examples[i] + 1, channel_names[channels[j]]))
                axes[j, i].plot(self.freq[0, 0], self.PSD[examples[i], channels[j]])
                axes[j, i].set_xlabel('Frequency (HZ)')
                axes[j, i].set_ylabel('PSD')
