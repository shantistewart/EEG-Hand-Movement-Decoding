from os import path
import numpy as np
import sys


# Keep in mind, the sampling rate is 250Hz.
# Input Patient number: 1-9
# Outputs:
#   left_array, right_array
#       size: (num_trials, num_channels, num_samples)
def ReadComp4(patient_num, path_to_file):
    if patient_num < 0 or patient_num > 9:
        sys.exit("Patient number invalid")

    left_file_form = path_to_file + "Data_Left_" + str(patient_num) + "_"
    right_file_form = path_to_file + "Data_Right_" + str(patient_num) + "_"
    unlabeled_file_form = path_to_file + "Data_Unlabeled_" + str(patient_num) + "_"

    left_array = []
    right_array = []
    unlabeled_array = []
    for i in range(1, 6):
        for j in range(1, 81):
            left_file = left_file_form + str(i) + "_" + str(j) + ".txt"
            right_file = right_file_form + str(i) + "_" + str(j) + ".txt"
            if path.exists(left_file):
                left_array += [np.loadtxt(left_file)]
            #else:
            #    print("Left file is not available %d %d %d" %(patient_num, i, j))
            if path.exists(right_file):
                right_array += [np.loadtxt(right_file)]
            #else:
            #    print("Right file is not available %d %d %d" %(patient_num, i, j))
        for l in range(1, 161):
            unlabeled_file = unlabeled_file_form + str(i) + "_" + str(l) + ".txt"
            if path.exists(unlabeled_file):
                unlabeled_array += [np.loadtxt(unlabeled_file)]
    left_array = np.array(left_array)
    right_array = np.array(right_array)
    unlabeled_array = np.array(unlabeled_array)
    left_array = np.transpose(left_array, (0, 2, 1))
    right_array = np.transpose(right_array, (0, 2, 1))
    unlabeled_array = np.transpose(unlabeled_array, (0, 2, 1))

    return left_array, right_array, unlabeled_array

# Stride and window are in seconds! Note that
# stride is the start time for a particular window.
# The window_len is the length of the window
# Output:
#   A 3D array of dimension: (strides, channels, window_len)
def stride_window(eeg_trial, stride, window_len, frequency):
    if window_len > eeg_trial.shape[1]:
        sys.exit("Window length longer than trial length")

    stride_examples = int(stride * frequency)
    window_examples = int(window_len * frequency)
    num_strides = eeg_trial.shape[1] // stride_examples
    # If window size is too long, need to cut back on number of stride
    while (num_strides * stride_examples) + window_examples > eeg_trial.shape[1]:
        num_strides = num_strides - 1

    grouped_features = []
    for i in range(num_strides + 1):
        window_start = i * stride_examples
        window_end = window_start + window_examples
        grouped_features += [eeg_trial[:, window_start:window_end]]
    return np.array(grouped_features)
