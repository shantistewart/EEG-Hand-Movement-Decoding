# This file contains a function to read data from data files.


import sys
from os import path
import numpy as np


# Function description: reads EEG data from data files and constructs data arrays.
# Inputs:
#   patient_num = number of patient
#   path_to_file = relative path to data files
# Outputs:
#   left_array = 3D array of left-hand data
#       size: (num_trials, num_channels, num_samples)
#   right_array = 3D array of right-hand data
#       size: (num_trials, num_channels, num_samples)
def ReadComp4(patient_num, path_to_file):
    if patient_num < 0 or patient_num > 9:
        sys.exit("Patient number invalid")

    left_file_form = path_to_file + "Data_Left_" + str(patient_num) + "_"
    right_file_form = path_to_file + "Data_Right_" + str(patient_num) + "_"

    left_array = []
    right_array = []
    for i in range(1, 6):
        for j in range(1, 81):
            left_file = left_file_form + str(i) + "_" + str(j) + ".txt"
            right_file = right_file_form + str(i) + "_" + str(j) + ".txt"
            if path.exists(left_file):
                left_array += [np.loadtxt(left_file)]
            if path.exists(right_file):
                right_array += [np.loadtxt(right_file)]
    left_array = np.array(left_array)
    right_array = np.array(right_array)
    left_array = np.transpose(left_array, (0, 2, 1))
    right_array = np.transpose(right_array, (0, 2, 1))

    return left_array, right_array
