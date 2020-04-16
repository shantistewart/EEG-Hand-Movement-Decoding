# This file contains unit tests for power_spectral_density.py (in feature_calculation folder):


import numpy as np
from feature_calculation import power_spectral_density as power


# --------------------TESTING estimate_psd() FUNCTION--------------------
print("\n----------TESTING estimate_psd()----------\n")

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

# call function:
Rxx, freq, PSD = power.estimate_psd(X, sample_freq)

# display autocorrelation functions (non-negative parts):
print("Autocorrelation functions (non-negative parts):\nSize: ", end="")
print(Rxx.shape)
print(Rxx)
print("")
# display power spectral density frequencies:
print("Power spectral densitiy frequencies:\nSize: ", end="")
print(freq.shape)
print(freq)
print("")
# display power spectral density values:
print("Power spectral density values:\nSize: ", end="")
print(PSD.shape)
print(PSD)
print("\n")
