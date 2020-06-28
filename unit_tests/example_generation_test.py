# This file contains unit tests for example_generation.py (in classifiers folder):


import numpy as np
import sklearn
from models.data_gathering import data4_reader
from models.classifiers import example_generation as example


print("\n")

# NOT TO BE MODIFIED:
# path to data files:
path_to_file = "../MATLAB/biosig/Data_txt/"
# sampling frequency:
sample_frequency = 250
# class labels:
LEFT_HAND_LABEL = 0
RIGHT_HAND_LABEL = 1

# subject number:
subject_num = 1

# dimensions of test arrays:
num_examples = 3
number_channels = 2
num_samples = 2
# test window and stride sizes:
window_size_ = 3
stride_size_ = 2
print("Window and stride sizes in seconds: ({0}, {1})\n".format(window_size_, stride_size_))


"""
# --------------------TESTING ReadComp4() FUNCTION--------------------
print("\n----------TESTING ReadComp4() FUNCTION----------\n")

# get raw data:
left_X, right_X = data4_reader.ReadComp4(subject_num, path_to_file)
# display shape of leftX and rightX:
print("leftX size: ", end="")
print(left_X.shape)
print("rightX size: ", end="")
print(right_X.shape)
print("\n")
"""


"""
# --------------------TESTING window_data() FUNCTION--------------------
print("\n----------TESTING window_data() FUNCTION----------\n")

# test array:
x = np.zeros((num_examples, number_channels, num_samples))
for i in range(num_examples):
    for j in range(number_channels):
        for k in range(num_samples):
            x[i, j, k] = (i + 1) * (k + 1) * (2 * np.mod(j + 1, 2) - 1)
print("Test input array:\nSize: ", end="")
print(x.shape)
print(x)
print("")

# call function:
x_window = example.window_data(x, window_size_, stride_size_)

# display windowed array:
print("Windowed array:\nSize: ", end="")
print(x_window.shape)
print(x_window)
print("\n")
"""


# """
# --------------------TESTING generate_examples() FUNCTION--------------------
print("\n----------TESTING generate_examples() FUNCTION----------\n")


# Function description: generates examples (raw data + class labels) with shuffled trials and sliding window
#   augmentation; slightly modified for testing.
# Inputs:
#   leftX = left hand raw data -- for testing
#       size: (0.5*num_trials, num_channels, num_samples)
#   rightX = right hand raw data -- for testing
#       size: (0.5*num_trials, num_channels, num_samples)
#   window_size = size of sliding window (to create more examples), in seconds
#   stride_size = size of "stride" of sliding window (to create more examples), in seconds
#   sample_freq = sampling frequency
# Outputs:
#   X_window = windowed raw data, with shuffled trials
#       size: (num_trials * num_windows, num_channels, window_size)
#   Y_window = class labels, with shuffled trials
#       size: (num_trials * num_windows, )
def generate_examples(leftX, rightX, window_size, stride_size, sample_freq):
    # convert window and stride sizes from seconds to samples:
    window_size = int(np.floor(sample_freq * window_size))
    stride_size = int(np.floor(sample_freq * stride_size))

    num_channels = leftX.shape[1]
    # generate corresponding class labels:
    leftY = LEFT_HAND_LABEL * np.ones(leftX.shape[0], dtype=int)
    rightY = RIGHT_HAND_LABEL * np.ones(rightX.shape[0], dtype=int)

    # concatenate left and right raw data/class labels:
    #   size of X: (num_trials, num_channels, num_samples), size of Y: (num_trials, )
    X = np.concatenate((leftX, rightX))
    Y = np.concatenate((leftY, rightY))
    print("Concatenated X:\nSize: ", end="")
    print(X.shape)
    print(X)
    print("")
    print("Concatenated Y:\nSize: ", end="")
    print(Y.shape)
    print(Y)
    print("")

    # shuffle raw data trials and class labels in unison:
    X, Y = sklearn.utils.shuffle(X, Y, random_state=0)
    print("Shuffled X:\nSize: ", end="")
    print(X.shape)
    print(X)
    print("")
    print("Shuffled Y:\nSize: ", end="")
    print(Y.shape)
    print(Y)
    print("")

    # create more examples by sliding window segmentation:
    #   size of X_window: (num_trials, num_windows, num_channels, window_size)
    X_window = example.window_data(X, window_size, stride_size)
    num_windows = X_window.shape[1]
    print("Window and stride sizes in samples: {0}, {1}\n".format(window_size, stride_size))
    print("Windowed X (with 4 dimensions):\nSize: ", end="")
    print(X_window.shape)
    print(X_window)
    print("")

    # combine first 2 dimensions of X_window:
    #   new size of X_window: (num_trials * num_windows, num_channels, window_size):
    X_window = np.reshape(X_window, (-1, num_channels, window_size))
    # expand class labels to match X_window:
    #   new size of Y_window: (num_trials * num_windows, )
    Y_window = np.repeat(Y, num_windows)

    return X_window, Y_window


# sampling frequency for testing:
sample_frequency = 1

# test arrays:
leftx = np.zeros((num_examples, number_channels, num_samples))
for i in range(num_examples):
    for j in range(number_channels):
        for k in range(num_samples):
            leftx[i, j, k] = -1 * (i + 1) * (j * num_samples + k + 1)
print("leftx:\nSize: ", end="")
print(leftx.shape)
print(leftx)
print("")
rightx = np.zeros((num_examples, number_channels, num_samples))
for i in range(num_examples):
    for j in range(number_channels):
        for k in range(num_samples):
            rightx[i, j, k] = (i + 1) * (j * num_samples + k + 1)
print("rightx:\nSize: ", end="")
print(rightx.shape)
print(rightx)
print("\n")

# call function:
x_window, y_window = generate_examples(leftx, rightx, window_size_, stride_size_, sample_frequency)

# display generated examples:
print("\nGenerated example data:\nSize: ", end="")
print(x_window.shape)
print(x_window)
print("")
print("Generated example class labels:\nSize: ", end="")
print(y_window.shape)
print(y_window)
print("\n")
# """


# """
# --------------------TESTING split_data() FUNCTION--------------------
print("\n----------TESTING split_data() FUNCTION----------\n")

# data set split parameters:
val_fract = 0.15
test_fract = 0.15

# call function:
X_train, Y_train, X_val, Y_val, X_test, Y_test = example.split_data(x_window, y_window, val_fract, test_fract)

# display training/validation/test sets features and class labels:
print("X_train:\nSize: ", end="")
print(X_train.shape)
print(X_train)
print("\nY_train:\nSize: ", end="")
print(Y_train.shape)
print(Y_train)
print("\nX_val:\nSize: ", end="")
print(X_val.shape)
print(X_val)
print("\nY_val:\nSize: ", end="")
print(Y_val.shape)
print(Y_val)
print("\nX_test:\nSize: ", end="")
print(X_test.shape)
print(X_test)
print("\nY_test:\nSize: ", end="")
print(Y_test.shape)
print(Y_test)
print("\n")
# """


# --------------------TESTING standardize_data() FUNCTION--------------------
print("\n----------TESTING standardize_data() FUNCTION----------\n")

# test arrays:
X_train = np.zeros((num_examples, number_channels, num_samples))
for i in range(num_examples):
    for j in range(number_channels):
        for k in range(num_samples):
            X_train[i, j, k] = (i + 1) * (j * num_samples + 1)
print("X_train:\nSize: ", end="")
print(X_train.shape)
print(X_train)
print("")
X_val = np.zeros((num_examples, number_channels, num_samples))
for i in range(num_examples):
    for j in range(number_channels):
        for k in range(num_samples):
            X_val[i, j, k] = (i + 1) * (j * num_samples + k + 1)
print("X_val:\nSize: ", end="")
print(X_val.shape)
print(X_val)
print("")
X_test = np.zeros((num_examples, number_channels, num_samples))
for i in range(num_examples):
    for j in range(number_channels):
        for k in range(num_samples):
            X_test[i, j, k] = 0.5 * (i + 1) * (j * num_samples + k + 1)
print("X_test:\nSize: ", end="")
print(X_test.shape)
print(X_test)
print("")

# call function:
X_train, X_val, X_test = example.standardize_data(X_train, X_val, X_test)

# display training/validation/test sets:
print("Standardized X_train:\nSize: ", end="")
print(X_train.shape)
print(X_train)
print("\nStandardized X_val:\nSize: ", end="")
print(X_val.shape)
print(X_val)
print("\nStandardized X_test:\nSize: ", end="")
print(X_test.shape)
print(X_test)
print("\n")
