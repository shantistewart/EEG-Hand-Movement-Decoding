from models.feature_calculation import feature_algorithms as feature
from models.data_gathering import data4_reader as feature_gen
import numpy as np
import tensorflow as tf
import sklearn

# This code groups features and labels, and
# translates the given data into separate
# windows and strides that can later be
# processed using principal component
# analysis.
# Input: Takes a list of numpy arrays where each array
#        consists of trials that are all associated with
#        the same class (left hand movement or right
#        hand movement.
# Output:
def group_feat_and_labels(window_len, stride, frequency, sorted_data, trial_label):
    feat_array = []
    label_array = []

    for index, trial_data in enumerate(sorted_data):
        grouped_features = feature_gen.stride_window(trial_data, stride, window_len, frequency)
        example_labels = [trial_label] * len(grouped_features)
        # place series of window/stride organized data together
        # this is an array of 2d arrays. This 2d array is the
        # matrix data
        feat_array += [grouped_features]
        label_array += [example_labels]

    return feat_array, label_array

def gather_shuffle_data(patient_num, path_to_file, window_len, stride, frequency):
    left_data, right_data = feature_gen.ReadComp4(patient_num, path_to_file)
    all_feature_data = []
    all_label_data = []

    # Go through left data and add features and labels
    for left_example in left_data:
        left_features, left_labels = group_feat_and_labels(window_len, stride, frequency, left_example, 0)
        # concatenate both arrays
        all_feature_data += left_features
        all_label_data += left_data

    # Go through right data and add features and labels
    for right_example in right_data:
        right_features, right_labels = group_feat_and_labels(window_len, stride, frequency, right_example, 1)
        # concatenate both arrays
        all_feature_data += right_features
        all_label_data += right_data

    # Shuffle data
    shuffled_features, shuffled_labels = sklearn.utils.shuffle(all_feature_data, all_label_data)

    return shuffled_features, shuffled_labels

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

