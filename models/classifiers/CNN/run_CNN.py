# This file contains code to train and evaluate a convolutional neural network for multiple subjects.


import numpy as np
import matplotlib.pyplot as plotter
from models.classifiers.CNN import evaluate_CNN


print("\n")

# NOT TO BE MODIFIED:
# sampling frequency:
sample_freq = 250

# subjects to evaluate:
subject_nums = np.array([1, 4, 5, 6, 7, 8, 9])

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
max_freq = 46.0
num_bins = 47
PCA = 0
# for CNN architecture:
num_conv_layers = 1
num_dense_layers = 2
num_kernels = 12
kernel_size = 3
pool_size = 2
num_hidden_nodes = 130
reg_type = 1
L2_reg = 0.032
dropout_reg = 0.4
# for training CNN:
num_epochs = 75
batch_size = 64
plot_learn_curve = False

# train and evaluate CNN:
avg_train, avg_val, avg_test, train, val, test = evaluate_CNN.train_eval_CNN(subject_nums, window_size_example,
                                                                             stride_size_example, sample_freq,
                                                                             num_conv_layers, num_dense_layers,
                                                                             num_kernels, kernel_size, pool_size,
                                                                             num_hidden_nodes, num_epochs, batch_size,
                                                                             window_size_PSD, stride_size_PSD, max_freq,
                                                                             num_bins, PCA=PCA, val_fract=val_fract,
                                                                             test_fract=test_fract, standard=standard,
                                                                             reg_type=reg_type, L2_reg=L2_reg,
                                                                             dropout_reg=dropout_reg, test=True,
                                                                             plot_learn_curve=plot_learn_curve)

# display training/validation/test accuracies:
print("\n\n\nTraining accuracies for subjects:")
print(train)
print("Average training accuracy: {0}\n".format(avg_train))
print("Validation accuracies for subjects:")
print(val)
print("Average validation accuracy: {0}\n".format(avg_val))
print("Test accuracies for subjects:")
print(test)
print("Average test accuracy: {0}\n".format(avg_test))

# plot a bar graph of accuracies:
evaluate_CNN.plot_accuracies(train, val, test)

# display all plots:
plotter.show()
