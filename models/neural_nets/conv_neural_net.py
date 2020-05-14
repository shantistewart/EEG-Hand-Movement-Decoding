# This file contains a class for a convolutional neural network.


import numpy as np
import matplotlib.pyplot as plotter
import tensorflow as tf
from tensorflow import keras


# Hyperparameters:
# network architecture:
num_conv_layers = 1
num_pool_layers = 1
num_connect_layers = 1
num_hidden_nodes = 50
# training:
num_epochs = 15
batch_size = 128
