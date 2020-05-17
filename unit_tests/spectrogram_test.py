# This file contains unit tests for spectrograms.py (in feature_calculation folder):


import numpy as np
from models.feature_calculation import spectrogram


# --------------------TESTING window_data() FUNCTION--------------------
print("\n----------TESTING window_data() FUNCTION----------\n")

# dimensions of test array:
num_examples = 3
num_channels = 2
num_samples = 10
# test array:
X = np.zeros((num_examples, num_channels, num_samples))
for i in range(num_examples):
    for j in range(num_channels):
        for k in range(num_samples):
            X[i, j, k] = (i+1) * (j*num_samples + k + 1)
print("Test input array:\nSize: ", end="")
print(X.shape)
print(X)
print("")
# test window and stride sizes:
window_size = 5
stride_size = 2
# number of chunks:
num_chunks = 2
print("Window size, stride size, number of chunks: ({0}, {1}, {2})\n".format(window_size, stride_size, num_chunks))

# call function:
X_window = spectrogram.window_data(X, window_size, stride_size, num_chunks)

# display windowed array:
print("Windowed array:\nSize: ", end="")
print(X_window.shape)
print(X_window)
print("\n")
