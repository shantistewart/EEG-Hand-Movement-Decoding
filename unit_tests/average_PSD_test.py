# This file contains unit tests for average_PSD.py (in feature_calculation folder):


import numpy as np
import matplotlib.pyplot as plotter
from feature_calculation import average_PSD


# --------------------TESTING calc_average_PSD() FUNCTION--------------------
print("\n----------TESTING calc_average_PSD() FUNCTION----------\n")

# test sampling frequency:
sample_freq = 100
# test selected frequency bins:
bins = np.array([[4.9, 5.1], [14.7, 14.7], [40.1, 49.9], [2.5, 47.5]])
print("Test frequency bins:\nSize: ", end="")
print(bins.shape)
print(bins)
print("")
# dimensions of test array:
num_examples = 2
num_channels = 3
num_samples = 10
# test input array:
PSD = np.zeros((num_examples, num_channels, num_samples))
for i in range(num_examples):
    for j in range(num_channels):
        for k in range(num_samples):
            PSD[i, j, k] = (i+1) * (j*num_samples + k + 1)
print("Test input array:\nSize: ", end="")
print(PSD.shape)
print(PSD)
print("")

# call function:
PSD_avg = average_PSD.calc_average_PSD(PSD, bins, sample_freq)

# display average PSD values:
print("Average PSD values:\nSize: ", end="")
print(PSD_avg.shape)
print(PSD_avg)
print("\n")


# --------------------TESTING plot_average_PSD() FUNCTION--------------------
print("\n----------TESTING plot_average_PSD() FUNCTION----------\n")

# examples to plot:
examples = np.array([0, 1])
# channels to plot:
channels = np.array([0, 2])
# names of all channels:
channel_names = np.array(['C1', 'C2', 'C3'])

# call function:
average_PSD.plot_average_PSD(PSD_avg, bins, examples, channels, channel_names)
# show plots:
plotter.show()
