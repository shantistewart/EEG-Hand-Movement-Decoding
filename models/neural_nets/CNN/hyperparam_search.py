# This file contains functions to tune the hyperparameters of a convolutional neural network.


import numpy as np
from models.neural_nets.CNN import evaluate_CNN


print("\n")

# subjects to evaluate:
subject_nums = np.array([1, 3, 5, 7, 9])

# call function:
train_acc, val_acc = evaluate_CNN.train_eval_CNN(subject_nums)
print("Training accuracies for subjects:")
print(train_acc)
print("")
print("Validation accuracies for subjects:")
print(val_acc)
