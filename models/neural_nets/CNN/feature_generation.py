# This file contains functions to generate features for neural networks.


import numpy as np
from models.data_gathering import data4_reader
from models.feature_calculation import spectrogram
from models.feature_calculation import feature_algorithms as feature

# class labels (do not modify!):
LEFT_HAND_LABEL = 0
RIGHT_HAND_LABEL = 1


# Function description: generates examples (raw data + class labels).
# Inputs:
#   subject_num = number of human subject (1-9)
#   path_to_file = path to data file
#   window_size = size of sliding window to create more examples, in seconds
#   stride_size = size of "stride" of sliding window to create more examples, in seconds
#   sample_freq = sampling frequency
# Outputs:
def generate_examples(subject_num, path_to_file, window_size, stride_size, sample_freq):
    # get data:
    leftX, rightX = data4_reader.ReadComp4(subject_num, path_to_file)

    # create more examples by sliding time window segmentation:
    #   new size: (num_examples, num_windows, num_channels, window_size)
    leftX = spectrogram.window_data(leftX, window_size, stride_size)
    rightX = spectrogram.window_data(rightX, window_size, stride_size)

    # combine first 2 dimensions (num_examples * num_windows):
    #   new size: (num_examples * num_windows, num_channels, window_size):
    leftX = np.reshape(leftX, (-1, leftX.shape[2], leftX.shape[3]))
    rightX = np.reshape(rightX, (-1, rightX.shape[2], rightX.shape[3]))

    # generate class labels:
    leftY = LEFT_HAND_LABEL * np.ones(leftX.shape[0], dtype=int)
    rightY = RIGHT_HAND_LABEL * np.ones(rightX.shape[0], dtype=int)

    # concatenate left and right raw data/class labels:
    X = np.concatenate((leftX, rightX))
    Y = np.concatenate((leftY, rightY))
