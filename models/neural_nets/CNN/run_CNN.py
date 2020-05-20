# This file contains code to run a convolutional neural network.


import numpy as np
import matplotlib.pyplot as plotter
from models.neural_nets import example_generation
from models.feature_calculation import feature_algorithms
from models.neural_nets.CNN import conv_neural_net

print("\n")


# NOT TO BE MODIFIED:
# path to data files:
path_to_data_file = "../../../MATLAB/biosig/Data_txt/"
# sampling frequency:
sample_freq = 250


# subject number:
subject_num = 1

# HYPERPARAMETERS:
# test set fraction:
test_fract = 0.2
# for creating more training examples:
window_size_example = 2.5
stride_size_example = 0.1
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
batch_size = 32
validation_fract = 0.2


# get data and generate examples:
X, Y = example_generation.generate_examples(subject_num, path_to_data_file, window_size_example, stride_size_example,
                                            sample_freq)
# display dimensions of raw data:
print("Size of raw data set: ", end="")
print(X.shape)

# generate spectrogram features:
X_spectro = feature_algorithms.spectrogram_algorithm(X, window_size_PSD, stride_size_PSD, sample_freq, max_freq,
                                                     num_bins, PCA, num_pcs, matrix_type, small_param)
# move channels axis to last:
X_spectro = np.transpose(X_spectro, axes=(0, 2, 3, 1))

# split features and class labels into training (+ validation) and test sets:
X_train, Y_train, X_test, Y_test = example_generation.split_train_test(X_spectro, Y, test_fract)
print("Size of train set: ", end="")
print(X_train.shape)
print("Size of test set: ", end="")
print(X_test.shape)

# create ConvNet object:
CNN = conv_neural_net.ConvNet(num_conv_layers, num_dense_layers, num_kernels, kernel_size, pool_size, num_hidden_nodes)

# build CNN model:
input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
CNN.build_model(input_shape)
# display model architecture:
print("\n")
CNN.model.summary()

# train model:
CNN.train_model(X_train, Y_train, num_epochs, batch_size, validation_fract)

# evaluate model:
test_acc = CNN.test_model(X_test, Y_test)

# plot learning curve:
CNN.plot_learn_curve()
plotter.show()
