# This file contains functions to generate training/validation/test examples for training and evaluation.


import numpy as np
import sklearn
from models.data_reading import data_reader


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


# Function description: generates examples (raw data + class labels) with shuffled trials and sliding window
#   augmentation.
# Inputs:
#   subject_num = number of subject (1-9)
#   path_to_file = path to data file
#   window_size = size of sliding window (to create more examples), in seconds
#   stride_size = size of "stride" of sliding window (to create more examples), in seconds
#   sample_freq = sampling frequency
# Outputs:
#   X_window = windowed raw data, with shuffled trials
#       size: (num_trials * num_windows, num_channels, window_size)
#   Y_window = class labels, with shuffled trials
#       size: (num_trials * num_windows, )
def generate_examples(subject_num, path_to_file, window_size, stride_size, sample_freq):
    # convert window and stride sizes from seconds to samples:
    window_size = int(np.floor(sample_freq * window_size))
    stride_size = int(np.floor(sample_freq * stride_size))

    # get raw data:
    #   size of leftX, rightX: (0.5*num_trials, num_channels, num_samples)
    leftX, rightX = data_reader.ReadComp4(subject_num, path_to_file)
    num_channels = leftX.shape[1]
    # generate corresponding class labels:
    leftY = LEFT_HAND_LABEL * np.ones(leftX.shape[0], dtype=int)
    rightY = RIGHT_HAND_LABEL * np.ones(rightX.shape[0], dtype=int)

    # concatenate left and right raw data/class labels:
    #   size of X: (num_trials, num_channels, num_samples), size of Y: (num_trials, )
    X = np.concatenate((leftX, rightX))
    Y = np.concatenate((leftY, rightY))

    # shuffle raw data trials and class labels in unison:
    X, Y = sklearn.utils.shuffle(X, Y, random_state=0)

    # create more examples by sliding window segmentation:
    #   size of X_window: (num_trials, num_windows, num_channels, window_size)
    X_window = window_data(X, window_size, stride_size)
    num_windows = X_window.shape[1]

    # combine first 2 dimensions of X_window:
    #   new size of X_window: (num_trials * num_windows, num_channels, window_size):
    X_window = np.reshape(X_window, (-1, num_channels, window_size))
    # expand class labels to match X_window:
    #   new size of Y_window: (num_trials * num_windows, )
    Y_window = np.repeat(Y, num_windows)

    return X_window, Y_window


# Function description: splits data set (features + class labels) into training, validation, and test sets, and shuffles
#   each set separately.
# Inputs:
#   X = features
#       size: (num_examples,...)
#   Y = class labels
#       size: (num_examples, )
#   val_fract = fraction of data to use as validation set
#   test_fract = fraction of data to use as test set
# Outputs:
#   X_train = (shuffled) training set features
#       size: ( (1-val_fract-test_fract) * num_examples,...)
#   Y_train = (shuffled) training set class labels
#       size: ( (1-val_fract-test_fract) * num_examples, )
#   X_val = (shuffled) validation set features
#       size: (val_fract * num_examples,...)
#   Y_val = (shuffled) validation set class labels
#       size: (val_fract * num_examples, )
#   X_test = (shuffled) test set features
#       size: (test_fract * num_examples,...)
#   Y_test = (shuffled) test set class labels
#       size: (test_fract * num_examples, )
def split_data(X, Y, val_fract, test_fract):
    # total number of examples:
    num_examples = X.shape[0]

    # start indices of validation and test sets:
    val_start_index = int(np.floor((1 - val_fract - test_fract) * num_examples))
    test_start_index = int(np.floor((1 - test_fract) * num_examples))

    # split data into training, validation, and test sets:
    X_train = X[:val_start_index]
    Y_train = Y[:val_start_index]
    X_val = X[val_start_index:test_start_index]
    Y_val = Y[val_start_index:test_start_index]
    X_test = X[test_start_index:]
    Y_test = Y[test_start_index:]

    # separately shuffle training, validation, and test sets:
    X_train, Y_train = sklearn.utils.shuffle(X_train, Y_train)
    X_val, Y_val = sklearn.utils.shuffle(X_val, Y_val)
    X_test, Y_test = sklearn.utils.shuffle(X_test, Y_test)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


# Function description: standardizes training set and standardizes validation and test sets with statistics from
#   training set.
#   standardization: remove mean (center) and divide by standard deviation (normalize)
# Inputs:
#   X_train = training set features
#       size: (num_train_examples,...)
#   X_val = validation set features
#       size: (num_val_examples,...)
#   X_test = test set features
#       size: (num_test_examples,...)
# Outputs:
#   X_train = standardized training set features
#       size: (num_train_examples,...)
#   X_val = standardized (by training set) validation set features
#       size: (num_val_examples,...)
#   X_test = standardized (by training set) test set features
#       size: (num_test_examples,...)
def standardize_data(X_train, X_val, X_test):
    # calculate statistics (across examples) of training set:
    mean_train = np.mean(X_train, axis=0, keepdims=True)
    std_train = np.std(X_train, axis=0, keepdims=True)

    # standardize training, validation, and test sets by training set statistics:
    X_train = (X_train - mean_train) / std_train
    X_val = (X_val - mean_train) / std_train
    X_test = (X_test - mean_train) / std_train

    return X_train, X_val, X_test
