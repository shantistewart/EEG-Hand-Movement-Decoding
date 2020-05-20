# This file contains code to run a convolutional neural network.


import numpy as np
# import function modules:
from models.neural_nets.CNN import example_generation
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
num_outputs = 1
num_conv_layers = 2
num_dense_layers = 1
num_filters = 3
kernel_size = 3
pool_size = 2
num_hidden_nodes = 200
# for training CNN:
num_epochs = 100
batch_size = 32
validation_fraction = 0.2


X, Y = example_generation.generate_examples(subject_num, path_to_data_file, window_size_example, stride_size_example,
                                            sample_freq)
# display dimensions of raw data:
print("Size of raw data: ", end="")
print(X.shape)

# generate spectrogram features:
X_spectro = feature_algorithms.spectrogram_algorithm(X, window_size_PSD, stride_size_PSD, sample_freq, max_freq,
                                                     num_bins, PCA, num_pcs, matrix_type, small_param)
# move channels axis to last:
X_spectro = np.transpose(X_spectro, axes=(0, 2, 3, 1))
# display dimensions of spectrogram features:
print("Size of spectrogram features: ", end="")
print(X_spectro.shape)
print("")

# build CNN model:
input_shape = (X_spectro.shape[1], X_spectro.shape[2], X_spectro.shape[3])
model = conv_neural_net.build_model(input_shape, num_outputs, num_conv_layers, num_dense_layers, num_filters,
                                    kernel_size, pool_size, num_hidden_nodes)
# display model architecture:
model.summary()
# train model:
history = conv_neural_net.train_model(model, X_spectro, Y, num_epochs, validation_fraction)
