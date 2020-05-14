# This file contains a class for a convolutional neural network.


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
    # output layer:
    model.add(layers.Dense(num_outputs))

    return model


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
