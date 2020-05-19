from models.feature_calculation import feature_algorithms as feature
from models.data_gathering import data4_reader as feature_gen
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plotter
import sklearn

# Labels (Do not modify)
########################
Left_Hand_Label = 0
Right_Hand_Label = 1
########################
# Other important variables (Do not modify)
########################
# path to data files:
path_to_data_file = "../../MATLAB/biosig/Data_txt/"
# frequency is 250Hz
sample_frequency = 250
# range of patients:
patient_range = range(1, 10)


########################

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

    for trial_data in sorted_data:
        # grouped features is a 3D array, (strides, windows, channels)
        # we wish to remove the stride dimension and toss all of the
        # stride examples into more (window, channel) arrays.
        grouped_features = feature_gen.stride_window(trial_data, stride, window_len, frequency)
        # place series of window/stride organized data together
        # this is an array of 2d arrays. This 2d array is the
        # matrix data
        if feat_array == []:
            feat_array = grouped_features
        else:
            feat_array = np.concatenate((feat_array, grouped_features), axis=0)

    label_array = [trial_label] * feat_array.shape[0]

    return feat_array, np.array(label_array)


def gather_shuffle_data(patient_num, path_to_file, window_len, stride, frequency):
    left_data, right_data = feature_gen.ReadComp4(patient_num, path_to_file)

    # Go through left data and add features and labels
    left_features, left_labels = group_feat_and_labels(window_len, stride, frequency, left_data, Left_Hand_Label)
    # concatenate both arrays
    all_feature_data = left_features
    all_label_data = left_labels

    # Go through right data and add features and labels
    right_features, right_labels = group_feat_and_labels(window_len, stride, frequency, right_data, Right_Hand_Label)
    # concatenate both arrays
    all_feature_data = np.concatenate((all_feature_data, right_features), axis=0)
    all_label_data = np.concatenate((all_label_data, right_labels), axis=0)

    #all_feature_data = np.swapaxes(all_feature_data, 1, 2)

    # Shuffle data
    shuffled_features, shuffled_labels = sklearn.utils.shuffle(all_feature_data, all_label_data)

    return np.array(shuffled_features), np.array(shuffled_labels)


# This function is work in progress... The hope is that
# this function will help rotate which data is for training,
# which is for testing, and which is for validation
def separate_data(train_st_index, val_st_index, test_st_index, data_features, data_labels):
    count = train_st_index
    stop = (train_st_index - 1) % len(data_labels)
    train_data = []
    train_labels = []
    val_data = []
    val_labels = []
    test_data = []
    test_labels = []
    # while count != stop:

    return train_data, train_labels, val_data, val_labels, test_data, test_labels


# Here we generate our TensorFlow model for our data
# This code is intended to fit the model, nothing more
# Output: TensorFlow model
def train_mlp(train_data, train_labels, bins, hid_layer_nodes=None, epoch_cnt=None):
    if epoch_cnt is None:
        epoch_cnt = 100

    if hid_layer_nodes is None:
        hid_layer_nodes = 30

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(3, bins.shape[0])),
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

    history = model.fit(train_data, train_labels, epochs=epoch_cnt, verbose=0)

    return model, history


def gather_train_test(patient_num, window_len, stride):
    shuffled_features, shuffled_labels = gather_shuffle_data(patient_num, path_to_data_file, window_len, stride,
                                                             sample_frequency)

    # Split the data up for training, validation, and testing
    ###########################################
    training_cutoff = int(percentage_training * len(shuffled_features))
    validation_cutoff = int(training_cutoff + percentage_validation * len(shuffled_features))
    testing_cutoff = training_cutoff + validation_cutoff

    train_data = shuffled_features[0:training_cutoff]
    train_labels = shuffled_labels[0:training_cutoff]
    val_data = shuffled_features[training_cutoff:validation_cutoff]
    val_labels = shuffled_labels[training_cutoff:validation_cutoff]
    test_data = shuffled_features[validation_cutoff:]
    test_labels = shuffled_labels[validation_cutoff:]

    # Process the data
    ###########################################

    processed_train = feature.average_PSD_algorithm(X=train_data, sample_freq=sample_frequency, bins=freq_bins)
    processed_val = feature.average_PSD_algorithm(X=val_data, sample_freq=sample_frequency, bins=freq_bins)
    processed_test = feature.average_PSD_algorithm(X=test_data, sample_freq=sample_frequency, bins=freq_bins)

    return processed_train, train_labels, processed_val, val_labels, processed_test, test_labels


if __name__ == '__main__':
    # Do not change:
    ############################
    percentage_training = 0.6
    percentage_validation = 0.2
    percentage_testing = 0.2
    ############################

    # Hyper Parameters
    ############################
    # set window length to be 1 second
    window_len = 1
    # set stride to be .5 seconds
    stride = .5
    # frequency bins
    freq_bins = np.array([[6., 8.], [8., 10.], [10., 12.], [12., 14.], [14., .5 * sample_frequency]])
    ############################

    patient_val = []

    for pat in patient_range:
        train_x, train_label, val_x, val_label, test_x, test_label = gather_train_test(pat, window_len,
                                                                                       stride)
        model, history = train_mlp(train_x, train_label, bins=freq_bins, hid_layer_nodes=20, epoch_cnt=90)
        val_loss, val_accuracy = model.evaluate(val_x, val_label)
        patient_val += [val_accuracy]

    patient_val = np.array(patient_val)
    average = np.average(patient_val)
    print("Average validation accuracy across patients: %.4f" % average)
    patient = []
    for pat in patient_range:
        patient += ['Patient #%d' % pat]

# patients 4, 6, 7, 8, 9 produce errors
    plotter.bar(patient, patient_val)
    ax = plotter.ylim((0.4, 1))
    plotter.ylabel('Validation Accuracy')
    plotter.title('Validation Accuracies for Patients')
    plotter.show()
