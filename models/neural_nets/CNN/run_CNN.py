# This file contains code to run a convolutional neural network.


import numpy as np
# import function modules:
from models.neural_nets.CNN import example_generation
from models.feature_calculation import feature_algorithms

print("\n")


# NOT TO BE MODIFIED:
# path to data files:
path_to_data_file = "../../../MATLAB/biosig/Data_txt/"
# sampling frequency:
sample_freq = 250


# subject number:
subject_num = 1

# HYPERPARAMETERS:
# for creating more training examples:
window_size_example = 3.0
stride_size_example = 0.1
# for spectrogram creation:
window_size_PSD = 1.0
stride_size_PSD = 0.1
max_freq = 25.0
num_bins = 25
PCA = 0
num_pcs = num_bins
matrix_type = 0
small_param = 0.0001


X, Y = example_generation.generate_examples(subject_num, path_to_data_file, window_size_example, stride_size_example,
                                            sample_freq)
# display dimensions of raw data:
print("Size of raw data: ", end="")
print(X.shape)

# generate spectrogram features:
X_spectro = feature_algorithms.spectrogram_algorithm(X, window_size_PSD, stride_size_PSD, sample_freq, max_freq,
                                                     num_bins, PCA, num_pcs, matrix_type, small_param)
# display dimensions of spectrogram features:
print("Size of spectrogram features: ", end="")
print(X_spectro.shape)
