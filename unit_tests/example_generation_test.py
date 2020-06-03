# This file contains unit tests for example_generation.py (in neural_nets folder):


import numpy as np
from models.neural_nets import example_generation as example
from models.data_gathering import data4_reader


print("\n")


# NOT TO BE MODIFIED:
# path to data files:
path_to_file = "../MATLAB/biosig/Data_txt/"
# sampling frequency:
sample_freq = 250

# subject number:
subject_num = 2

# temporary testing:
# get raw data:
leftX, rightX = data4_reader.ReadComp4(subject_num, path_to_file)
# display shape of leftX and rightX:
print("leftX size: ", end="")
print(leftX.shape)
print("rightX size: ", end="")
print(rightX.shape)
print("")


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
X_window = example.window_data(X, window_size, stride_size)

# display windowed array:
print("Windowed array:\nSize: ", end="")
print(X_window.shape)
print(X_window)
print("")
