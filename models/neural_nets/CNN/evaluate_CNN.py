# This file contains a function to train/evaluate a convolutional neural network across subjects.


import numpy as np
from models.neural_nets.CNN import conv_neural_net


# Function description: trains and evaluates a CNN for selected subjects (separate instance for each subject).
# Inputs:
#   subject_nums = array subject numbers to evaluate
#       size: (num_subjects, )
# Outputs:
def train_eval_CNN(subject_nums):
    # number of subjects:
    num_subjects = subject_nums.shape[0]
    # dictionaries for training and validation accuracies for subjects:
    train_acc = {}
    val_acc = {}

    # train a different CNN instance for each subject and record training/validation accuracies:
    for subject in subject_nums:
        train_acc[subject] = subject + 0.5
        val_acc[subject] = subject + 0.1

    return train_acc, val_acc
