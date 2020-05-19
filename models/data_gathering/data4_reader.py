import numpy as np
import sys


# Keep in mind, the sampling rate is 250Hz.
# Input Patient number: 1-9
def ReadComp4(patient_num, path_to_file):
    if patient_num < 0 or patient_num > 9:
        sys.exit("Patient number invalid")

    left_file_form = path_to_file + "Data_Left_" + str(patient_num) + "_"
    right_file_form = path_to_file + "Data_Right_" + str(patient_num) + "_"

    left_array = []
    right_array = []

    for i in range(1, 3):
        for j in range(1, 61):
            left_file = left_file_form + str(i) + "_" + str(j) + ".txt"
            right_file = right_file_form + str(i) + "_" + str(j) + ".txt"
            left_array += [np.loadtxt(left_file)]
            right_array += [np.loadtxt(right_file)]

    return np.array(left_array), np.array(right_array)

# Stride and window are in seconds! Note that
# stride is the start time for a particular window.
# The window_len is the length of the window
# Output:
#   A 3D array of dimension: (strides, channels, window_len)
def stride_window(eeg_trial, stride, window_len, frequency):
    if window_len > len(eeg_trial):
        sys.exit("Window length longer than trial length")

    stride_examples = int(stride * frequency)
    window_examples = int(window_len * frequency)
    num_strides = len(eeg_trial) // stride_examples
    # If window size is too long, need to cut back on number of stride
    while (num_strides * stride_examples) + window_examples > len(eeg_trial):
        num_strides = num_strides - 1

    grouped_features = []
    for i in range(num_strides):
        window_start = i * stride_examples
        window_end = window_start + window_examples
        grouped_features += [eeg_trial[window_start:window_end]]
    return np.transpose(np.array(grouped_features), (0, 2, 1))

