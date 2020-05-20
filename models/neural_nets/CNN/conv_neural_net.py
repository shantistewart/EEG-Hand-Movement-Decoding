# This file contains functions to build and train a convolutional neural network.


import numpy as np
import matplotlib.pyplot as plotter
import tensorflow as tf
from tensorflow.keras import layers, models
from models.neural_nets import example_generation
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
#   plot_learn_curve(): plots learning curve (training and validation accuracy vs. epochs).
class ConvNet:
    """Class for a convolutional neural network for binary classification."""

    # Constructor:
    def __init__(self, num_conv_layers, num_dense_layers, num_kernels, kernel_size, pool_size, num_hidden_nodes):
        self.model = None
        self.history = None
        self.num_conv_layers = num_conv_layers
        self.num_dense_layers = num_dense_layers
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.num_hidden_nodes = num_hidden_nodes

    # Methods:

    # Function description: builds CNN architecture.
    # Inputs:
    #   input_shape = dimensions of input features
    # Outputs: None
    def build_model(self, input_shape):
        self.model = models.Sequential()

        # convolutional and max pooling layers:
        self.model.add(layers.Conv2D(self.num_kernels, self.kernel_size, activation='relu', input_shape=input_shape))
        self.model.add(layers.MaxPool2D(self.pool_size))
        self.model.add(layers.Conv2D(self.num_kernels, self.kernel_size, activation='relu'))
        self.model.add(layers.MaxPool2D(self.pool_size))

        # fully connected layers:
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(self.num_hidden_nodes, activation='relu'))
        # output layer:
        self.model.add(layers.Dense(1, activation='sigmoid'))

    # Function description: compiles and trains CNN.
    # Inputs:
    #   X_train = training set features
    #   Y_train = training set class labels
    #   num_epochs = number of epochs to train for
    #   batch_size = mini-batch size for training
    #   validation_fract = fraction of training set to use as validation set
    # Outputs: none
    def train_model(self, X_train, Y_train, num_epochs=10, batch_size=32, validation_fract=0.2):
        # compile model:
        self.model.compile(optimizer='Adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                           metrics=['binary_accuracy'])

        # train model:
        self.history = self.model.fit(X_train, Y_train, epochs=int(num_epochs), verbose=2,
                                      validation_split=validation_fract)

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
    # Inputs: none
    # Outputs: none
    def plot_learn_curve(self):
        # extract training and validation accuracies:
        train_acc = self.history.history['binary_accuracy']
        val_acc = self.history.history['val_binary_accuracy']

        # plot learning curve:
        fig, axes = plotter.subplots()
        axes.set_title('CNN with {0} Convolutional Layers and {1} Dense Layers'
                       .format(self.num_conv_layers, self.num_dense_layers))
        axes.plot(train_acc, label='training accuracy')
        axes.plot(val_acc, label='validation accuracy')
        axes.set_xlabel('Epochs')
        axes.set_ylabel('Accuracy')
        axes.legend(loc='center right')

    test_fract = 0.2
    # for spectrogram creation:
    window_size_PSD = 0.8
    stride_size_PSD = 0.05
    max_freq = 25.0
    num_bins = 50
    PCA = 0
    num_pcs = num_bins
    matrix_type = 0
    small_param = 0.0001

    # Function description: generates training and test set features for CNN.
    # Inputs:
    #   X = 3D array of raw signal values for multiple channels for multiple examples
    #       size: (num_examples, num_channels, num_samples)
    #   Y = class labels
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
    #   test_fract = fraction of data to use as test set
    # Outputs:
    def generate_features(self, X, Y, window_size, stride_size, sample_freq, max_freq, num_bins, PCA=0, num_pcs=None,
                          matrix_type=0, small_param=0.0001, test_fract=0.2):
        # generate spectrogram features:
        X_spectro = feature_algorithms.spectrogram_algorithm(X, window_size, stride_size, sample_freq, max_freq,
                                                             num_bins, PCA=PCA, num_pcs=num_pcs,
                                                             matrix_type=matrix_type, small_param=small_param)
        # move channels axis to last (for compatibility with CNN architecture):
        X_spectro = np.transpose(X_spectro, axes=(0, 2, 3, 1))

        # split features and class labels into training (+ validation) and test sets:
        X_train, Y_train, X_test, Y_test = example_generation.split_train_test(X_spectro, Y, test_fract=test_fract)

        return X_train, Y_train, X_test, Y_test

