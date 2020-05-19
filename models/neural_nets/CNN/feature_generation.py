# This file contains functions to generate features for neural networks.


import numpy as np
from models.data_gathering import data4_reader
from models.feature_calculation import spectrogram
from models.feature_calculation import feature_algorithms as feature


# Function description: performs a sliding-window segmentation on raw signal values.
# Inputs:
#   subject_num = number of human subject (1-9)
#   path_to_file = path to data file
#   window_size = size of sliding window to create more examples, in seconds
#   stride_size = size of "stride" of sliding window to create more examples, in seconds
#   sample_freq = sampling frequency
# Outputs:
def generate_examples(subject_num, path_to_file, window_size, stride_size, sample_freq):
    # get data:
    left_array, right_array = data4_reader.ReadComp4(subject_num, path_to_file)
