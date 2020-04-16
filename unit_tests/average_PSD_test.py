# This file contains unit tests for average_PSD.py (in feature_calculation folder):


import numpy as np
from feature_calculation import average_PSD


# --------------------TESTING average_PSD() FUNCTION--------------------
print("\n----------TESTING average_PSD() FUNCTION----------\n")

# test sampling frequency:
sample_freq = 100
# dimensions of test array:
num_examples = 2
num_channels = 3
num_samples = 5
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