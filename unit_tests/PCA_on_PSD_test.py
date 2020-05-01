# This file contains unit tests for PCA_on_PSD.py (in feature_calculation folder):


import numpy as np
import matplotlib.pyplot as plotter
from feature_calculation import PCA_on_PSD as PCA


# --------------------TESTING log_normalize() FUNCTION--------------------
print("\n----------TESTING log_normalize() FUNCTION----------\n")

# dimensions of test array:
num_examples = 3
num_channels = 2
num_freq = 4
# test input array:
PSD = np.zeros((num_examples, num_channels, num_freq))
for i in range(num_examples):
    for j in range(num_channels):
        for k in range(num_freq):
            PSD[i, j, k] = (i % 2 + 1) * (j * num_freq + k + 1)
print("Test input array:\nSize: ", end="")
print(PSD.shape)
print(PSD)
print("")
# small parameter to avoid ln(0):
small_param = 0.0

# call function:
PSD_norm = PCA.log_normalize(PSD, small_param)

"""
# display log-normalized PSD values:
print("Log-normalized PSD values:\nSize: ", end="")
print(PSD_norm.shape)
print(PSD_norm)
print("")
"""


"""
# --------------------TESTING unnorm_correlation() FUNCTION--------------------
print("\n----------TESTING unnorm_correlation() FUNCTION----------\n")

# call function:
corr_matrices = PCA.unnorm_correlation(PSD)

print("Channel-specific autocorrelation matrices (unnormalized):\nSize: ", end="")
print(corr_matrices.shape)
print(corr_matrices)
"""


# --------------------TESTING unnorm_covariance() FUNCTION--------------------
print("\n----------TESTING unnorm_covariance() FUNCTION----------\n")

# call function:
cov_matrices = PCA.unnorm_covariance(PSD)

print("Channel-specific autocovariance matrices (unnormalized):\nSize: ", end="")
print(cov_matrices.shape)
print(cov_matrices)
