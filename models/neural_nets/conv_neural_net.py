# This file contains a class for a convolutional neural network.


import numpy as np
import matplotlib.pyplot as plotter
import tensorflow as tf
from tensorflow.keras import layers, models


def build_model(input_shape, num_outputs, num_conv_layers=2, num_dense_layers=1, num_filters=2, kernel_size=3,
                pool_size=2, num_hidden_nodes=50):
    model = models.Sequential()

    # convolutional and max pooling layers:
    model.add(layers.Conv2D(num_filters, kernel_size, padding='valid', activation='relu', input_shape=input_shape))
    model.add(layers.MaxPool2D(pool_size, padding='valid'))
    model.add(layers.Conv2D(num_filters, kernel_size, padding='valid', activation='relu', input_shape=input_shape))
    model.add(layers.MaxPool2D(pool_size, padding='valid'))

    # fully connected layers:
    model.add(layers.Flatten())
    model.add(layers.Dense(num_hidden_nodes, activation='relu'))
    model.add(layers.Dense(num_outputs))

    return model
