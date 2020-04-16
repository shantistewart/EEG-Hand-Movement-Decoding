# This file contains unit tests for the RawPSD class in RawPSD_class.py (in classes folder):


import numpy as np
import matplotlib.pyplot as plotter
# import function modules:
from feature_calculation import power_spectral_density as power
from classes import RawPSD_class
from feature_calculation import average_PSD


# --------------------TESTING RawPSD CLASS--------------------
print("\n----------TESTING RawPSD CLASS----------\n")

# dimensions of test array:
num_examples = 3
num_channels = 4
num_samples = 100
# test sampling frequency:
sample_freq = 20
# sampling period:
sample_period = 1 / sample_freq
# sample points:
time = np.arange(0, num_samples)
# test array:
X = np.zeros((num_examples, num_channels, num_samples))
for i in range(num_examples):
    for j in range(num_channels):
        # frequency of cosine wave (Hz):
        freq_cosine = (i+1)
        X[i, j] = np.cos((2 * np.pi * freq_cosine) * sample_period * time) + np.random.uniform(-.2*(j+1), .2*(j+1), num_samples)

# estimate power spectral density:
Rxx, freq, PSD = power.estimate_psd(X, sample_freq)
# create Raw_PSD object:
raw_psd_object = RawPSD_class.RawPSD(num_examples, num_channels, num_samples, sample_freq, X, Rxx, freq, PSD)

# examples to plot:
examples = np.array([0, 2])
# channels to plot:
channels = np.array([0, 3])
# names of all channels:
channel_names = np.array(['C1', 'C2', 'C3', 'C4'])
# display selected raw signals:
raw_psd_object.plot_raw_signal(examples, channels, channel_names)
# display selected autocorrelation functions:
raw_psd_object.plot_autocorr(examples, channels, channel_names)
# display selected power spectral densities:
raw_psd_object.plot_PSD(examples, channels, channel_names)

# test selected frequency bins:
bins = np.array([[0, 2], [2, 4], [4, 6], [6, 8], [8, 10]])
# calculate average PSD values in selected frequency bins:
PSD_avg = average_PSD.average_PSD(PSD, bins, sample_freq)
# plot bar graph:
average_PSD.plot_average_PSD(PSD_avg, bins, examples, channels, channel_names)

# show plots:
plotter.show()
