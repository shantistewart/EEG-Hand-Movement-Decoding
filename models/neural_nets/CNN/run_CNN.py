# This file contains code to run a convolutional neural network for binary classification.


import matplotlib.pyplot as plotter
from models.neural_nets import example_generation
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
# for data set creation:
window_size_example = 2.5
stride_size_example = 0.25
val_fract = 0.15
test_fract = 0.15
standard = True
# for spectrogram creation:
window_size_PSD = 0.8
stride_size_PSD = 0.05
max_freq = 25.0
num_bins = 50
PCA = 0
num_pcs = num_bins
matrix_type = 2
small_param = 0.0001
# for CNN architecture:
num_conv_layers = 2
num_dense_layers = 1
num_kernels = 3
kernel_size = 3
pool_size = 2
num_hidden_nodes = 200
reg_type = 1
L2_reg = 0.005
dropout_reg = 0.0
# for training CNN:
num_epochs = 100
batch_size = 64

# get data and generate examples:
X, Y = example_generation.generate_examples(subject_num, path_to_data_file, window_size_example, stride_size_example,
                                            sample_freq)
# display dimensions of raw data:
print("Size of raw data set: ", end="")
print(X.shape)

# create ConvNet object:
CNN = conv_neural_net.ConvNet(num_conv_layers, num_dense_layers, num_kernels, kernel_size, pool_size, num_hidden_nodes)

# generate training and test features:
X_train, Y_train, X_val, Y_val, X_test, Y_test = CNN.generate_features(X, Y, window_size_PSD, stride_size_PSD,
                                                                       sample_freq, max_freq, num_bins, PCA=PCA,
                                                                       num_pcs=num_pcs, matrix_type=matrix_type,
                                                                       small_param=small_param, val_fract=val_fract,
                                                                       test_fract=test_fract, standard=standard)
print("Size of training set: ", end="")
print(X_train.shape)
print("Size of validation set: ", end="")
print(X_val.shape)
print("Size of test set: ", end="")
print(X_test.shape)

# build CNN model:
input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
CNN.build_model(input_shape, reg_type=reg_type, L2_reg=L2_reg, dropout_reg=dropout_reg)
# display model architecture:
print("\n")
CNN.model.summary()

# train model:
CNN.train_model(X_train, Y_train, X_val, Y_val, num_epochs, batch_size)

# evaluate model:
test_acc = CNN.test_model(X_test, Y_test)

# plot learning curve:
CNN.plot_learn_curve(subject_num)
plotter.show()
