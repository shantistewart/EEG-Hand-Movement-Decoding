# This file contains functions to build and train a convolutional neural network.


import matplotlib.pyplot as plotter
import tensorflow as tf
from tensorflow.keras import layers, models


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

