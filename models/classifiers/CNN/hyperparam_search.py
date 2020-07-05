# This file contains code to tune the hyperparameters of a convolutional neural network across multiple subjects.


import numpy as np
import matplotlib.pyplot as plotter
from models.classifiers.CNN import evaluate_CNN


print("\n")

# NOT TO BE MODIFIED:
# sampling frequency:
sample_freq = 250

# subjects to evaluate:
subject_nums = np.array([1, 4, 5, 6, 7, 8, 9])
# number of hyperparameter search iterations:
num_iterations = 50

# constant hyperparameters:
val_fract = 0.15
test_fract = 0.15
standard = True
PCA = 0
num_epochs = 50
batch_size = 64

# HYPERPARAMETERS RANGES:
# for data set creation:
window_size_example_range = [2.5, 2.5]
stride_size_example_range = [0.25, 0.25]
# for spectrogram creation:
window_size_PSD = 0.8
stride_size_PSD_range = [0.05, 0.05]
max_freq_range = [30.0, 50.0]
num_bins_range = [40, 50]
# for CNN architecture:
num_conv_layers_range = [1, 2]
num_dense_layers_range = [1, 2]
num_kernels_range = [10, 20]
kernel_size = 3
pool_size = 2
num_hidden_nodes_range = [100, 200]
reg_type_range = [1, 2]
L2_reg_range = [0.01, 0.05]
dropout_reg_range = [0.2, 0.5]


# initialize rolling best hyperparameter combination (max average validation accuracy across subjects):
best_avg_train_acc = 0.0
best_avg_val_acc = 0.0
best_train_acc = None
best_val_acc = None
best_window_size_example = None
best_stride_size_example = None
best_window_size_PSD = None
best_stride_size_PSD = None
best_max_freq = None
best_num_bins = None
best_num_conv_layers = None
best_num_dense_layers = None
best_num_kernels = None
best_num_hidden_nodes = None
best_reg_type = None
best_L2_reg = None
best_dropout_reg = None

# perform a random hyperparameter search:
for i in range(num_iterations):
    print("\n\n\n--------------------ITERATION {0} (OUT OF {1})--------------------\n".format(i+1, num_iterations))

    # randomly select hyperparameter values:

    # for data set creation:
    window_size_example = (window_size_example_range[1] - window_size_example_range[0]) * np.random.rand() + \
                          window_size_example_range[0]
    stride_size_example = (stride_size_example_range[1] - stride_size_example_range[0]) * np.random.rand() + \
                          stride_size_example_range[0]

    # for spectrogram creation:
    stride_size_PSD = (stride_size_PSD_range[1] - stride_size_PSD_range[0]) * np.random.rand() + \
                      stride_size_PSD_range[0]
    max_freq = np.random.randint(max_freq_range[0], max_freq_range[1] + 1)
    num_bins = np.random.randint(num_bins_range[0], num_bins_range[1] + 1)

    # for CNN architecture:
    num_conv_layers = np.random.randint(num_conv_layers_range[0], num_conv_layers_range[1] + 1)
    num_dense_layers = np.random.randint(num_dense_layers_range[0], num_dense_layers_range[1] + 1)
    num_kernels = np.random.randint(num_kernels_range[0], num_kernels_range[1] + 1)
    num_hidden_nodes = np.random.randint(num_hidden_nodes_range[0], num_hidden_nodes_range[1] + 1)
    reg_type = np.random.randint(reg_type_range[0], reg_type_range[1] + 1)
    L2_reg = (L2_reg_range[1] - L2_reg_range[0]) * np.random.rand() + L2_reg_range[0]
    dropout_reg = (dropout_reg_range[1] - dropout_reg_range[0]) * np.random.rand() + dropout_reg_range[0]

    # display hyperparameter values:
    print("HYPERPARAMETERS:\n")
    print("window_size_example: {0}".format(window_size_example))
    print("stride_size_example: {0}".format(stride_size_example))
    print("window_size_PSD: {0}".format(window_size_PSD))
    print("stride_size_PSD: {0}".format(stride_size_PSD))
    print("max_freq: {0}".format(max_freq))
    print("num_bins: {0}".format(num_bins))
    print("num_conv_layers: {0}".format(num_conv_layers))
    print("num_dense_layers: {0}".format(num_dense_layers))
    print("num_kernels: {0}".format(num_kernels))
    print("num_hidden_nodes: {0}".format(num_hidden_nodes))
    print("reg_type: {0}".format(reg_type))
    print("L2_reg: {0}".format(L2_reg))
    print("dropout_reg: {0}\n".format(dropout_reg))

    # train and evaluate CNN:
    avg_train, avg_val, avg_test, train, val, test = evaluate_CNN.train_eval_CNN(subject_nums, window_size_example,
                                                                                 stride_size_example, sample_freq,
                                                                                 num_conv_layers, num_dense_layers,
                                                                                 num_kernels, kernel_size, pool_size,
                                                                                 num_hidden_nodes, num_epochs,
                                                                                 batch_size, window_size_PSD,
                                                                                 stride_size_PSD, max_freq, num_bins,
                                                                                 val_fract=val_fract,
                                                                                 test_fract=test_fract,
                                                                                 standard=standard, reg_type=reg_type,
                                                                                 L2_reg=L2_reg, dropout_reg=dropout_reg)

    # update rolling best hyperparameter combination:
    if avg_val > best_avg_val_acc:
        best_avg_train_acc = avg_train
        best_avg_val_acc = avg_val
        best_train_acc = train
        best_val_acc = val
        best_window_size_example = window_size_example
        best_stride_size_example = stride_size_example
        best_window_size_PSD = window_size_PSD
        best_stride_size_PSD = stride_size_PSD
        best_max_freq = max_freq
        best_num_bins = num_bins
        best_num_conv_layers = num_conv_layers
        best_num_dense_layers = num_dense_layers
        best_num_kernels = num_kernels
        best_num_hidden_nodes = num_hidden_nodes
        best_reg_type = reg_type
        best_L2_reg = L2_reg
        best_dropout_reg = dropout_reg
    # display rolling best hyperparameter combination:
    print("\n\nROLLING BEST HYPERPARAMETERS:\n")
    print("Average training accuracy: {0}".format(best_avg_train_acc))
    print("Average validation accuracy: {0}".format(best_avg_val_acc))
    print("window_size_example: {0}".format(best_window_size_example))
    print("stride_size_example: {0}".format(best_stride_size_example))
    print("window_size_PSD: {0}".format(best_window_size_PSD))
    print("stride_size_PSD: {0}".format(best_stride_size_PSD))
    print("max_freq: {0}".format(best_max_freq))
    print("num_bins: {0}".format(best_num_bins))
    print("num_conv_layers: {0}".format(best_num_conv_layers))
    print("num_dense_layers: {0}".format(best_num_dense_layers))
    print("num_kernels: {0}".format(best_num_kernels))
    print("num_hidden_nodes: {0}".format(best_num_hidden_nodes))
    print("reg_type: {0}".format(best_reg_type))
    print("L2_reg: {0}".format(best_L2_reg))
    print("dropout_reg: {0}\n".format(best_dropout_reg))


# display training and validation accuracies for subjects for best hyperparameter combination:
print("\n\nBest training accuracies for subjects:")
print(best_train_acc)
print("Best average training accuracy: {0}\n".format(best_avg_train_acc))
print("Best validation accuracies for subjects:")
print(best_val_acc)
print("Best average validation accuracy: {0}\n".format(best_avg_val_acc))

# plot a bar graph of accuracies:
evaluate_CNN.plot_accuracies(best_train_acc, best_val_acc)

# display plots:
plotter.show()
