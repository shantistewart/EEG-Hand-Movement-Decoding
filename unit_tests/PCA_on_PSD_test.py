# This file contains unit tests for PCA_on_PSD.py (in feature_calculation folder):


import numpy as np
import matplotlib.pyplot as plotter
from feature_calculation import PCA_on_PSD as PCA


# dimensions of test array:
num_examples = 3
num_channels = 2
num_freq = 5
# test input array:
PSD = np.zeros((num_examples, num_channels, num_freq))
for i in range(num_examples):
    for j in range(num_channels):
        for k in range(num_freq):
            PSD[i, j, k] = (i + 1) * (j * num_freq + k + 1)
            # PSD[i, j, k] = ((i % 2 + 1) * (j * num_freq + k + 1)) * (2*(k % 2) - 1)
        # PSD[i, j, :] = np.random.rand(num_freq)
print("\nTest input array:\nSize: ", end="")
print(PSD.shape)
print(PSD)
print("")
# small parameter to avoid ln(0):
small_param = 0.0001


"""
# --------------------TESTING log_normalize() FUNCTION--------------------
print("\n----------TESTING log_normalize() FUNCTION----------\n")

# call function:
PSD_norm = PCA.log_normalize(PSD, small_param)

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

# display unnormalized autocorrelation matrices:
print("Channel-specific autocorrelation matrices (unnormalized):\nSize: ", end="")
print(corr_matrices.shape)
print(corr_matrices)
print("")
"""


"""
# --------------------TESTING unnorm_covariance() FUNCTION--------------------
print("\n----------TESTING unnorm_covariance() FUNCTION----------\n")

# call function:
cov_matrices = PCA.unnorm_covariance(PSD)

# display unnormalized autocovariance matrices:
print("Channel-specific autocovariance matrices (unnormalized):\nSize: ", end="")
print(cov_matrices.shape)
print(cov_matrices)
print("")
"""


"""
# --------------------TESTING pearson_covariance() FUNCTION--------------------
print("\n----------TESTING pearson_covariance() FUNCTION----------\n")

# call function:
pearson_cov_matrices = PCA.pearson_covariance(PSD)

# display Pearson correlation coefficient autocovariance matrices:
print("Channel-specific Pearson autocovariance matrices:\nSize: ", end="")
print(pearson_cov_matrices.shape)
print(pearson_cov_matrices)
print("")
"""


"""
# --------------------TESTING calc_eig_vects() FUNCTION--------------------
print("\n----------TESTING calc_eig_vects() FUNCTION----------\n")

# dimensions of test array:
num_channels = 2
num_freq = 5
# test input array:
matrices = np.zeros((num_channels, num_freq, num_freq))
for n in range(num_channels):
    for i in range(num_freq):
        for j in range(num_freq):
            matrices[n, i, j] = (n + 1) * ((i + 1) * (j + 1))  # * (2*((i+j+1) % 2) - 1)
print("Test input matrices:\nSize: ", end="")
print(matrices.shape)
print(matrices)
print("")

# call function:
eig_vects = PCA.calc_eig_vects(matrices)

# display sorted eigenvectors:
print("Sorted eigenvectors:\nSize: ", end="")
print(eig_vects.shape)
print(eig_vects)
print("")
"""


# """
# --------------------TESTING project_onto_pcs() FUNCTION--------------------
print("\n----------TESTING project_onto_pcs() FUNCTION----------\n")

# test principal components array:
pcs = np.zeros((num_channels, num_freq, num_freq))
for n in range(num_channels):
    for i in range(num_freq):
        for j in range(num_freq):
            pcs[n, i, j] = j
print("Test principal components array:\nSize: ", end="")
print(pcs.shape)
print(pcs)
print("")
# number of principal components to project onto:
num_pcs = num_freq - 1
print("Number of principal components to project onto: {0}\n".format(num_pcs))

# call function:
project_weights = PCA.project_onto_pcs(PSD, pcs, num_pcs)

# display projection weights:
print("Projection weights:\nSize: ", end="")
print(project_weights.shape)
print(project_weights)
print("\n")
# """
