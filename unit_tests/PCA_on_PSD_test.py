# This file contains unit tests for PCA_on_PSD.py (in feature_calculation folder):


import numpy as np
import matplotlib.pyplot as plotter
from feature_calculation import PCA_on_PSD


# --------------------TESTING log_normalize() FUNCTION--------------------
print("\n----------TESTING log_normalize() FUNCTION----------\n")

# dimensions of test array:
num_examples = 3
num_channels = 2
num_samples = 4
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
