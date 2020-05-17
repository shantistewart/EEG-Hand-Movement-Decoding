# This file contains unit tests for spectrograms.py (in feature_calculation folder):


import numpy as np
from models.feature_calculation import spectrogram


# --------------------TESTING window_data() FUNCTION--------------------
print("\n----------TESTING window_data() FUNCTION----------\n")

# test window and stride sizes:
window_size = 5
stride_size = 2
# test number of chunks:
num_chunks = 2
print("Window size, stride size, number of chunks: ({0}, {1}, {2})\n".format(window_size, stride_size, num_chunks))
# dimensions of test array:
num_examples = 3
num_channels = 2
num_samples = 10
# test array:
X = np.zeros((num_examples, num_channels, num_samples))
for i in range(num_examples):
    for j in range(num_channels):
        for k in range(num_samples):
            X[i, j, k] = (i + 1) * (2*np.mod(j+1, 2) - 1) * (k + 1)
print("Test input array:\nSize: ", end="")
print(X.shape)
print(X)
print("")

# call function:
X_window = spectrogram.window_data(X, window_size, stride_size, num_chunks)

# display windowed array:
print("Windowed array:\nSize: ", end="")
print(X_window.shape)
print(X_window)
print("\n")


# --------------------TESTING create_spectrogram() FUNCTION--------------------
print("\n----------TESTING create_spectrogram() FUNCTION----------\n")

# test number of chunks:
num_chunks = 2
# test sampling frequency:
sample_freq = 100
# test max frequency and number of bins:
max_freq = 25
num_bins = 25
# test PCA parameters:
PCA = 0
num_pcs = 10
matrix_type = 0
small_param = 0.0001

# call function:
X_window = spectrogram.create_spectrogram(X_window, num_chunks, sample_freq, max_freq, num_bins, PCA, num_pcs, matrix_type, small_param)

# display partially flattened windowed array:
print("Partially flattened windowed array:\nSize: ", end="")
print(X_window.shape)
print(X_window)
print("\n")
