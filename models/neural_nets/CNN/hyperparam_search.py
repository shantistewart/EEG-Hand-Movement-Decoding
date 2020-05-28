# This file contains functions to tune the hyperparameters of a convolutional neural network.


import numpy as np
from models.neural_nets.CNN import evaluate_CNN


# subjects to evaluate:
subject_nums = np.array([1, 3, 5, 7, 9])

evaluate_CNN.train_eval_CNN(subject_nums)
