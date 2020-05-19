# This file contains code to run a convolutional neural network.


import numpy as np
# import function modules:
from models.neural_nets.CNN import example_generation
from models.feature_calculation import feature_algorithms


# NOT TO BE MODIFIED:
# path to data files:
path_to_data_file = "../../../MATLAB/biosig/Data_txt/"
# sampling frequency:
sample_freq = 250


# subject number:
subject_num = 1

# HYPERPARAMETERS:
window_size_example = 2.0
stride_size_example = 0.1

print("\n")


X, Y = example_generation.generate_examples(subject_num, path_to_data_file, window_size_example, stride_size_example,
                                            sample_freq)
# display dimensions of raw data:
print("Size of X: ", end="")
print(X.shape)
print("Size of Y: ", end="")
print(Y.shape)
