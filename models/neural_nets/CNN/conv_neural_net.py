# This file contains functions to build and train a convolutional neural network.


import matplotlib.pyplot as plotter
import tensorflow as tf
from tensorflow.keras import layers, models


# Function description: builds convolutional neural network architecture.
# Inputs:
#   input_shape = dimensions of input features
#   num_outputs = number of nodes in last layer
#   num_conv_layers = number of convolutional layers
#       a max pooling layer is added after each convolutional layer
#   num_dense_layers = number of fully connected layers after convolutional layers
#   num_filters = number of filters for convolutional layers
#   kernel_size = size of convolutional filters
#   pool_size = size of pooling filters for max pooling layers
#   num_hidden_nodes = number of nodes of fully connected layers (except last layer)
# Outputs:
#   model = TensorFlow Sequential model object
def build_model(input_shape, num_outputs=1, num_conv_layers=2, num_dense_layers=1, num_filters=3, kernel_size=3,
                pool_size=2, num_hidden_nodes=50):
    model = models.Sequential()

    # convolutional and max pooling layers:
    model.add(layers.Conv2D(num_filters, kernel_size, padding='valid', data_format='channels_first', activation='relu',
                            input_shape=input_shape))
    model.add(layers.MaxPool2D(pool_size, padding='valid', data_format='channels_first'))
    model.add(layers.Conv2D(num_filters, kernel_size, padding='valid', data_format='channels_first', activation='relu'))
    model.add(layers.MaxPool2D(pool_size, padding='valid', data_format='channels_first'))

    # fully connected layers:
    model.add(layers.Flatten())
    model.add(layers.Dense(num_hidden_nodes, activation='relu'))
    # output layer:
    model.add(layers.Dense(num_outputs, activation='sigmoid'))

    return model


# Function description: compiles and trains convolutional neural network.
# Inputs:
#   model = TensorFlow Sequential model object
#   x_train = training set features
#   y_train = training set class labels
#   num_epochs = number of epochs to train for
#   batch_size = mini-batch size for training
#   validation_fraction = fraction of training set to use as validation set
# Outputs:
#   history = TensorFlow history object
def train_model(model, x_train, y_train, num_epochs=10, batch_size=32, validation_fraction=0.15):
    # compile model:
    model.compile(optimizer='Adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # train model:
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, verbose=2,
                        validation_split=validation_fraction)
    # extract training and validation accuracies:
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    # display learning curve:
    plotter.plot(train_acc, label='train_accuracy')
    plotter.plot(val_acc, label='val_accuracy')
    plotter.xlabel('Epochs')
    plotter.ylabel('Accuracy')
    plotter.legend(loc='lower right')
    plotter.show()

    return history
