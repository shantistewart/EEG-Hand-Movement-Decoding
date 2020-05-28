# This file contains functions to generate training/test examples for neural networks.


import numpy as np
import sklearn
from models.data_gathering import data4_reader


# class labels (do not modify!):
LEFT_HAND_LABEL = 0
RIGHT_HAND_LABEL = 1


# Function description: performs sliding-window segmentation.
# Inputs:
#   X = 3D array of signal values for multiple channels for multiple examples
#       size: (num_examples, num_channels, num_samples)
#   window_size = size of sliding window, in samples
#   stride_size = size of sliding window "stride", in samples
# Outputs:
#   X_window = windowed raw signals
#       size: (num_examples, num_windows, num_channels, window_size)
def window_data(X, window_size, stride_size):
    # number of examples:
    num_examples = X.shape[0]
    # number of channels:
    num_channels = X.shape[1]
    # number of samples:
    num_samples = X.shape[2]

    # determine number of possible windows:
    num_windows = int(np.floor((num_samples - window_size) / stride_size) + 1)

    X_window = np.zeros((num_examples, num_windows, num_channels, window_size))
    for i in range(num_windows):
        # start of window index:
        start_index = i * stride_size
        # end of window index (inclusive):
        end_index = start_index + window_size - 1

        X_window[:, i, :, :] = X[:, :, start_index:end_index+1]

    return X_window


# Function description: generates examples (raw data + class labels) with sliding window segmentation.
# Inputs:
#   subject_num = number of human subject (1-9)
#   path_to_file = path to data file
#   window_size = size of sliding window to create more examples, in seconds
#   stride_size = size of "stride" of sliding window to create more examples, in seconds
#   sample_freq = sampling frequency
# Outputs:
#   X = raw data
#       size: (2 * num_trials * num_windows, num_channels, window_size)
#   Y = class labels
#       size: (2 * num_trials * num_windows, )
def generate_examples(subject_num, path_to_file, window_size, stride_size, sample_freq):
    # convert window and stride sizes from seconds to samples:
    window_size = int(np.floor(sample_freq * window_size))
    stride_size = int(np.floor(sample_freq * stride_size))

    # get data:
    #   size: (num_trials, num_channels, num_samples)
    leftX, rightX = data4_reader.ReadComp4(subject_num, path_to_file)

    # create more examples by sliding time window segmentation:
    #   new size: (num_trials, num_windows, num_channels, window_size)
    leftX = window_data(leftX, window_size, stride_size)
    rightX = window_data(rightX, window_size, stride_size)

    # combine first 2 dimensions (num_examples * num_windows):
    #   new size: (num_trials * num_windows, num_channels, window_size):
    leftX = np.reshape(leftX, (-1, leftX.shape[2], leftX.shape[3]))
    rightX = np.reshape(rightX, (-1, rightX.shape[2], rightX.shape[3]))

    # generate class labels:
    leftY = LEFT_HAND_LABEL * np.ones(leftX.shape[0], dtype=int)
    rightY = RIGHT_HAND_LABEL * np.ones(rightX.shape[0], dtype=int)

    # concatenate left and right raw data/class labels:
    X = np.concatenate((leftX, rightX))
    Y = np.concatenate((leftY, rightY))

    return X, Y


# Function description: shuffles data and splits features and class labels into training (+ validation) and test sets.
# Inputs:
#   X = features
#       size: (num_examples,...)
#   Y = class labels
#       size: (num_examples, )
#   test_fract = fraction of data to use as test set
# Outputs:
#   X_train = (shuffled) training set features
#       size: ((1-test_fract) * num_examples,...)
#   Y_train = (shuffled) training set class labels
#       size: ((1-test_fract) * num_examples, )
#   X_test = (shuffled) test set features
#       size: (test_fract * num_examples,...)
#   Y_test = (shuffled) test set class labels
#       size: (test_fract * num_examples, )
def split_train_test(X, Y, test_fract=0.2):
    # shuffle raw data and class labels in unison:
    X, Y = sklearn.utils.shuffle(X, Y, random_state=0)

    # total number of examples:
    num_examples = X.shape[0]

    # start index of test set:
    test_index = int(np.floor((1 - test_fract) * num_examples))

    # split data into training and test sets:
    X_train = X[:test_index]
    Y_train = Y[:test_index]
    X_test = X[test_index:]
    Y_test = Y[test_index:]

    return X_train, Y_train, X_test, Y_test


# Function description: standardizes training set and standardizes test set with statistics from training set.
#   standardization: remove mean (center) and divide by variance (normalize)
# Inputs:
#   X_train = training set features
#       size: (num_train_examples,...)
#   X_test = test set features
#       size: (num_test_examples,...)
# Outputs:
#   X_train = standardized training set features
#       size: (num_train_examples,...)
#   X_test = standardized (by training set) test set features
#       size: (num_test_examples,...)
def standardize_data(X_train, X_test):
    # calculate statistics (across examples) of training set:
    mean_train = np.mean(X_train, axis=0, keepdims=True)
    std_train = np.std(X_train, axis=0, keepdims=True)

    # standardize training and test sets by training set statistics:
    X_train = (X_train - mean_train) / std_train
    X_test = (X_test - mean_train) / std_train

    return X_train, X_test
