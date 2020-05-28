# This file contains functions to tune the hyperparameters of a convolutional neural network.


import numpy as np
from models.neural_nets.CNN import evaluate_CNN


print("\n")

# NOT TO BE MODIFIED:
# sampling frequency:
sample_freq = 250

# subjects to evaluate:
subject_nums = np.array([1, 3, 7])

# HYPERPARAMETERS:
# for data set creation:
window_size_example = 2.5
stride_size_example = 0.1
test_fract = 0.15
standard = True
# for spectrogram creation:
window_size_PSD = 0.8
stride_size_PSD = 0.05
max_freq = 25.0
num_bins = 50
PCA = 0
num_pcs = num_bins
matrix_type = 0
small_param = 0.0001
# for CNN architecture:
num_conv_layers = 2
num_dense_layers = 1
num_kernels = 3
kernel_size = 3
pool_size = 2
num_hidden_nodes = 200
# for training CNN:
num_epochs = 100
batch_size = 64
validation_fract = 0.2

# call function:
avg_train_acc, avg_val_acc, train_acc, val_acc = evaluate_CNN.train_eval_CNN(subject_nums, window_size_example,
                                                                             stride_size_example, sample_freq,
                                                                             num_conv_layers, num_dense_layers,
                                                                             num_kernels, kernel_size, pool_size,
                                                                             num_hidden_nodes, num_epochs, batch_size,
                                                                             validation_fract, window_size_PSD,
                                                                             stride_size_PSD, max_freq, num_bins,
                                                                             PCA=PCA, num_pcs=num_pcs,
                                                                             matrix_type=matrix_type,
                                                                             small_param=small_param,
                                                                             test_fract=test_fract, standard=standard)

# display training and validation accuracies for subjects:
print("Training accuracies for subjects:")
print(train_acc)
print("Average training accuracy: {0}\n".format(avg_train_acc))
print("Validation accuracies for subjects:")
print(val_acc)
print("Average validation accuracy: {0}\n".format(avg_val_acc))
