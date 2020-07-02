# This file contains unit tests for functions in spectrogram.py (models/feature_calculation/spectrogram.py).


import numpy as np
from models.classifiers import example_generation
from models.feature_calculation import spectrogram


# --------------------TESTING window_data() FUNCTION--------------------
print("\n----------TESTING window_data() FUNCTION----------\n")

# dimensions of test array:
num_examples = 2
num_channels = 3
num_samples = 11
# test array:
X = np.zeros((num_examples, num_channels, num_samples))
for i in range(num_examples):
    for j in range(num_channels):
        for k in range(num_samples):
            X[i, j, k] = (i + 1) * (k + 1) * (2*np.mod(j+1, 2) - 1)
print("Test input array:\nSize: ", end="")
print(X.shape)
print(X)
print("")
# test window and stride sizes:
window_size = 5
stride_size = 2
print("Window size and stride sizes: ({0}, {1})\n".format(window_size, stride_size))

# call function:
X_window = example_generation.window_data(X, window_size, stride_size)

# display windowed array:
print("Windowed array:\nSize: ", end="")
print(X_window.shape)
print(X_window)
print("")


# --------------------TESTING create_spectrogram() FUNCTION--------------------
print("\n----------TESTING create_spectrogram() FUNCTION----------\n")

# test sampling frequency:
sample_freq = 100
# test max frequency and number of bins:
max_freq = sample_freq / 4
num_bins = 2*window_size
# test PCA parameters:
PCA = 0
num_pcs = 10
matrix_type = 0
small_param = 0.0001

# call function:
spectrograms = spectrogram.create_spectrogram(X_window, sample_freq, max_freq, num_bins, PCA, num_pcs, matrix_type,
                                              small_param)

# display spectrograms:
print("Spectrograms:\nSize: ", end="")
print(spectrograms.shape)
print(spectrograms)
print("\n")
