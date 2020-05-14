from feature_calculation import feature_algorithms as feature
import numpy as np
import tensorflow as tf

# Here we have the hyper-parameters and the array of
# np training data as inputs.
# Output:
#
def train_mlp(window_len, stride, train_data, train_labels, bins, hid_layer_nodes=None, epoch_cnt=None):
    if epoch_cnt is None:
        epochs = 100

    if hid_layer_nodes is None:
        hid_layer_nodes = 30

    processed_train = feature.average_PSD_algorithm(train_data, bins)

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(3, 5)),
        tf.keras.layers.Dense(hid_layer_nodes, activation='relu'),
        tf.keras.layers.Dense(2)  # output layer
        # Activation of each layer is linear (i.e. no activation:
        # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.fit(train_data, train_labels, epochs=epoch_cnt)

    return model

