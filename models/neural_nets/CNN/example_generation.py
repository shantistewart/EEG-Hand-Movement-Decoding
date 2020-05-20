# This file contains functions to generate features for neural networks.


import numpy as np
import sklearn
# import function modules:
from models.data_gathering import data4_reader
from models.feature_calculation import spectrogram

# class labels (do not modify!):
LEFT_HAND_LABEL = 0
RIGHT_HAND_LABEL = 1


# Function description: generates examples (raw data + class labels) with sliding window segmentation and shuffling.
# Inputs:
#   subject_num = number of human subject (1-9)
#   path_to_file = path to data file
#   window_size = size of sliding window to create more examples, in seconds
#   stride_size = size of "stride" of sliding window to create more examples, in seconds
#   sample_freq = sampling frequency
# Outputs:
#   X = (shuffled) raw data
#       size: (2 * num_trials * num_windows, num_channels, window_size)
#   Y = (shuffled) class labels
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
    leftX = spectrogram.window_data(leftX, window_size, stride_size)
    rightX = spectrogram.window_data(rightX, window_size, stride_size)

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

    # shuffle raw data and class labels in unison:
    X, Y = sklearn.utils.shuffle(X, Y)

    return X, Y


# Function description: splits features and class labels into training (+ validation) and test sets.
# Inputs:
#   X = (shuffled) features
#       size: (num_examples,...)
#   Y = (shuffled) class labels
#       size: (num_examples,...)
#   test_fract = fraction of data to use as test set
# Outputs:
#   X_train = training set features
#   Y_train = training set class labels
#   X_test = test set features
#   Y_test = test set class labels
def split_train_test(X, Y, test_fract=0.2):
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
