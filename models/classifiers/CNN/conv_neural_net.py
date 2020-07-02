# This file contains a class to build, train, and evaluate a convolutional neural network.


import numpy as np
import matplotlib.pyplot as plotter
import tensorflow as tf
from tensorflow.keras import models, layers, regularizers
from models.classifiers import example_generation
from models.feature_calculation import feature_algorithms


# Class description: builds, trains, and evaluates a convolutional neural network for binary classification.
# Instance variables:
#   num_conv_layers = number of convolutional layers
#       a max pooling layer is added after each convolutional layer
#   num_dense_layers = number of fully connected layers after convolutional layers
#   num_kernels = number of kernels for convolutional layers
#   kernel_size = size of convolutional kernels
#   pool_size = size of pooling filters for max pooling layers
#   num_hidden_nodes = number of nodes of fully connected layers (except last layer)
# Inputs to constructor: all instance variables except model and history
# Methods:
#   build_model(): builds CNN architecture.
#   train_model(): compiles and trains CNN.
#   test_model(): evaluates CNN by computing accuracy on a test set.
#   plot_learn_curve(): plots learning curve (training and validation accuracy vs. epochs).
#   generate_features(): generates training and test set features for CNN.
class ConvNet:
    """Class for a convolutional neural network for binary classification."""

    def __init__(self, num_conv_layers, num_dense_layers, num_kernels, kernel_size, pool_size, num_hidden_nodes):
        self.model = None
        self.history = None
        self.num_conv_layers = num_conv_layers
        self.num_dense_layers = num_dense_layers
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.num_hidden_nodes = num_hidden_nodes

    # Function description: builds CNN architecture.
    # Inputs:
    #   input_shape = dimensions of input features
    #   reg_type = parameter to select type of regularization
    #       if reg_type == 1: apply L2 regularization
    #       else if reg_type == 2: apply dropout regularization
    #       else: don't apply regularization
    #   L2_reg = L2 regularization parameter
    #   dropout_reg = dropout regularization parameter
    # Outputs: None
    def build_model(self, input_shape, reg_type=1, L2_reg=0.0, dropout_reg=0.0):
        self.model = models.Sequential()

        # add convolutional and max pooling layers:
        self.model.add(layers.Conv2D(self.num_kernels, self.kernel_size, activation='relu', input_shape=input_shape))
        self.model.add(layers.MaxPool2D(self.pool_size))
        for _ in range(self.num_conv_layers - 1):
            self.model.add(layers.Conv2D(self.num_kernels, self.kernel_size, activation='relu'))
            self.model.add(layers.MaxPool2D(self.pool_size))

        self.model.add(layers.Flatten())

        # add fully connected layers:
        for _ in range(self.num_dense_layers):
            # include L2 regularization if selected:
            if reg_type == 1:
                self.model.add(layers.Dense(self.num_hidden_nodes, activation='relu',
                                            kernel_regularizer=regularizers.l2(L2_reg)))
            else:
                self.model.add(layers.Dense(self.num_hidden_nodes, activation='relu'))
            # include dropout regularization if selected:
            if reg_type == 2:
                self.model.add(layers.Dropout(dropout_reg))

        # output layer:
        if reg_type == 1:
            self.model.add(layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(L2_reg)))
        else:
            self.model.add(layers.Dense(1, activation='sigmoid'))

    # Function description: compiles and trains CNN.
    # Inputs:
    #   X_train = training set features
    #   Y_train = training set class labels
    #   X_val = validation set features
    #   Y_val = validation set class labels
    #   num_epochs = number of epochs to train for
    #   batch_size = mini-batch size for training
    # Outputs: none
    def train_model(self, X_train, Y_train, X_val, Y_val, num_epochs, batch_size):
        # compile model:
        self.model.compile(optimizer='Adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                           metrics=['binary_accuracy'])

        # train model:
        self.history = self.model.fit(x=X_train, y=Y_train, batch_size=int(batch_size), epochs=int(num_epochs),
                                      verbose=2, validation_data=(X_val, Y_val))

    # Function description: evaluates CNN by computing accuracy on a test set.
    # Inputs:
    #   X_test = test set features
    #   Y_test = test set class labels
    # Outputs:
    #   test_acc = test set accuracy
    def test_model(self, X_test, Y_test):
        print("\n")
        test_loss, test_acc = self.model.evaluate(X_test, Y_test, verbose=1)
        print("Test set accuracy: {0}\n".format(test_acc))

        return test_acc

    # Function description: plots learning curve (training and validation accuracy vs. epochs).
    # Inputs:
    #   subject_num = number of subject
    # Outputs: none
    def plot_learn_curve(self, subject_num):
        # extract training and validation accuracies:
        train_acc = self.history.history['binary_accuracy']
        val_acc = self.history.history['val_binary_accuracy']

        # plot learning curve:
        fig, axes = plotter.subplots()
        axes.set_title('Subject {0}: CNN with {1} Convolutional Layers and {2} Dense Layers'
                       .format(subject_num, self.num_conv_layers, self.num_dense_layers))
        axes.plot(train_acc, label='training accuracy')
        axes.plot(val_acc, label='validation accuracy')
        axes.set_xlabel('Epochs')
        axes.set_ylabel('Accuracy')
        axes.legend(loc='upper right')

    # Function description: generates training and test set features for CNN.
    # Inputs:
    #   X = windowed raw data, with shuffled trials
    #       size: (num_examples, num_channels, window_size)
    #   Y = class labels, with shuffled trials
    #       size: (num_examples, )
    #   window_size = size of sliding window to calculate PSD, in seconds
    #   stride_size = size of "stride" of sliding window to calculate PSD, in seconds
    #   sample_freq = sampling frequency
    #   max_freq = maximum frequency of PSD to consider
    #   num_bins = number of frequency bins for average PSD calculation
    #   PCA = parameter to select whether to apply PCA algorithm
    #       if PCA == 1: PCA algorithm is applied
    #       else: PCA algorithm is not applied
    #   num_pcs = number of principal components (eigenvectors) to project onto
    #       validity: num_pcs <= num_bins
    #   matrix_type = parameter to select which type of statistical matrix to calculate:
    #       if matrix type == 1: autocorrelation matrices are calculated
    #       if matrix type == 2: autocovariance matrices are calculated
    #       else: Pearson autocovariance matrices are calculated
    #   small_param = a small number to ensure that log(0) does not occur for log-normalization
    #   val_fract = fraction of data to use as validation set
    #   test_fract = fraction of data to use as test set
    #   standard = parameter to select whether to standardize features
    #       if standard == True: features are standardized
    #       else: features are not standardized
    # Outputs:
    #   X_train = (shuffled) training set features
    #       size: ( (1-val_fract-test_fract) * num_examples,...)
    #   Y_train = (shuffled) training set class labels
    #       size: ( (1-val_fract-test_fract) * num_examples, )
    #   X_val = (shuffled) validation set features
    #       size: (val_fract * num_examples,...)
    #   Y_val = (shuffled) validation set class labels
    #       size: (val_fract * num_examples, )
    #   X_test = (shuffled) test set features
    #       size: (test_fract * num_examples,...)
    #   Y_test = (shuffled) test set class labels
    #       size: (test_fract * num_examples, )
    def generate_features(self, X, Y, window_size, stride_size, sample_freq, max_freq, num_bins, PCA=0, num_pcs=None,
                          matrix_type=2, small_param=0.0001, val_fract=0.2, test_fract=0.15, standard=True):
        # generate spectrogram features:
        X_spectro = feature_algorithms.spectrogram_algorithm(X, window_size, stride_size, sample_freq, max_freq,
                                                             num_bins, PCA=PCA, num_pcs=num_pcs,
                                                             matrix_type=matrix_type, small_param=small_param)
        # move channels axis to last (for compatibility with CNN architecture):
        X_spectro = np.transpose(X_spectro, axes=(0, 2, 3, 1))

        # split features and class labels into training, validation, and test sets:
        X_train, Y_train, X_val, Y_val, X_test, Y_test = example_generation.split_data(X_spectro, Y, val_fract,
                                                                                       test_fract)

        # standardize features if selected:
        if standard:
            X_train, X_val, X_test = example_generation.standardize_data(X_train, X_val, X_test)

        return X_train, Y_train, X_val, Y_val, X_test, Y_test

