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

    for i in range(1, 2):
        for j in range(1, 61):
            left_file = left_file_form + str(i) + "_" + str(j) + ".txt"
            right_file = right_file_form + str(i) + "_" + str(j) + ".txt"
            left_array += [np.loadtxt(left_file)]
            right_array += [np.loadtxt(right_file)]

    return left_array, right_array

# Stride and window are in seconds! Note that
# stride is the start time for a particular window.
# The window_len is the length of the window
def stride_window(eeg_trial, stride, window_len):
    num_strides = len(eeg_trial) // stride
    grouped_features = []
    for i in range(num_strides):
        window_start = i * stride
        window_end = window_start + window_len
        grouped_features += [eeg_trial[window_start:window_end]]
    return grouped_features

