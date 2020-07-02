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
Unlabeled_Hand_Label = 2
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
    grouped_features = np.array([])
    last_array = np.array([])

    for trial_data in sorted_data:
        # grouped features is a 3D array, (strides, windows, channels)
        # we wish to remove the stride dimension and toss all of the
        # stride examples into more (window, channel) arrays.
        select_array = np.array([feature_gen.stride_window(trial_data, stride, window_len, frequency)])
        if grouped_features.size != 0:
            grouped_features = np.concatenate((grouped_features, select_array), axis=0)
        else:
            grouped_features = select_array
        # place series of window/stride organized data together
        # this is an array of 2d arrays. This 2d array is the
        # matrix data
        # if feat_array == []:
        #    feat_array = grouped_features
        # else:
        #    feat_array = np.concatenate((feat_array, grouped_features), axis=0)
    feat_array = np.array(grouped_features)
    label_array = [trial_label] * feat_array.shape[0]

    return feat_array, np.array(label_array)


def gather_shuffle_data(patient_num, path_to_file, window_len, stride, frequency):
    left_data, right_data, unlabeled_data = feature_gen.ReadComp4(patient_num, path_to_file)

    # Go through left data and add features and labels
    left_features, left_labels = group_feat_and_labels(window_len, stride, frequency, left_data, Left_Hand_Label)
    # concatenate both arrays
    all_feature_data = left_features
    all_label_data = left_labels
    print(left_features.shape)

    # Go through right data and add features and labels
    right_features, right_labels = group_feat_and_labels(window_len, stride, frequency, right_data, Right_Hand_Label)
    # concatenate both arrays
    all_feature_data = np.concatenate((all_feature_data, right_features), axis=0)
    all_label_data = np.concatenate((all_label_data, right_labels), axis=0)
    print(right_features.shape)

    # Go through unlabeled data and add features and labels
    """
    unlabeled_features, unlabeled_labels = group_feat_and_labels(window_len, stride, frequency, unlabeled_data,
                                                                 Unlabeled_Hand_Label)
    print(unlabeled_features.shape)
    all_feature_data = np.concatenate((all_feature_data, unlabeled_features), axis=0)
    all_label_data = np.concatenate((all_label_data, unlabeled_labels), axis=0)
    """
    return all_feature_data, all_label_data


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
def train_mlp(train_data, train_labels, bins, hid_layer_nodes=None, epoch_cnt=None, plot=False):
    if epoch_cnt is None:
        epoch_cnt = 100

    if hid_layer_nodes is None:
        hid_layer_nodes = 30

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(3, bins.shape[0])),
        tf.keras.layers.Dense(hid_layer_nodes, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')  # output layer
        # Activation of each layer is linear (i.e. no activation:
        # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    history = model.fit(train_data, train_labels, epochs=epoch_cnt, verbose=0,
                        )

    if plot is True:
        train_acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        # plot learning curve:
        fig, axes = plotter.subplots()
        axes.set_title('MLP Learning Plot')
        axes.plot(train_acc, label='training accuracy')
        axes.plot(val_acc, label='validation accuracy')
        axes.set_xlabel('Epochs')
        axes.set_ylabel('Accuracy')
        axes.legend(loc='center right')

    return model


def organize_by_windows(data_set, data_labels):
    new_labels = np.array([])
    data_set_extended = np.array([])
    for index, window_data in enumerate(data_set):
        # Make a new label array that uses the same label, but
        # repeated for each stride
        generated_labels = np.array([data_labels[index]] * data_set.shape[1])

        if data_set_extended.size == 0:
            data_set_extended = window_data
        else:
            data_set_extended = np.concatenate((np.array(data_set_extended), window_data), axis=0)
        new_labels = np.concatenate((new_labels, generated_labels))

    return data_set_extended, new_labels


def gather_train_test(patient_num, window_len, stride, freq_bins, percentage_training):
    shuffled_features, shuffled_labels = gather_shuffle_data(patient_num, path_to_data_file, window_len, stride,
                                                             sample_frequency)
    # Split the data up for training, validation, and testing
    ###########################################
    training_cutoff = int(percentage_training * len(shuffled_features))
    validation_cutoff = int(.8 * training_cutoff)

    train_data = shuffled_features[0:training_cutoff]
    train_labels = shuffled_labels[0:training_cutoff]
    # validation_data = shuffled_features[validation_cutoff:training_cutoff]
    # validation_labels = shuffled_labels[validation_cutoff:training_cutoff]
    test_data = shuffled_features[training_cutoff:]
    test_labels = shuffled_labels[training_cutoff:]

    # Remove one dimension of the array and concatenate windows
    # in particular datasets
    train_data, train_labels = organize_by_windows(train_data, train_labels)
    # validation_data, validation_labels = organize_by_windows(validation_data, validation_labels)
    test_data, test_labels = organize_by_windows(test_data, test_labels)

    # Shuffle the data:
    train_data, train_labels = sklearn.utils.shuffle(train_data, train_labels)
    # validation_data, validation_labels = sklearn.utils.shuffle(validation_data, validation_labels)
    test_data, test_labels = sklearn.utils.shuffle(test_data, test_labels)

    # Process the data
    ###########################################

    processed_train = feature.average_PSD_algorithm(X=train_data, sample_freq=sample_frequency, bins=freq_bins)
    # processed_val = feature.average_PSD_algorithm(X=validation_data, sample_freq=sample_frequency, bins=freq_bins)
    processed_test = feature.average_PSD_algorithm(X=test_data, sample_freq=sample_frequency, bins=freq_bins)

    return processed_train, train_labels, processed_test, test_labels


def play_hyperparam(win, stride, num_bin, percen_train):
    bin_width = 50.0 / num_bin
    freq_bins = np.zeros((num_bin, 2))
    for i in range(num_bin):
        freq_bins[i, 0] = i * bin_width
        freq_bins[i, 1] = (i + 1) * bin_width
    ############################

    patient_test = []
    patient_label = []

    for pat in patient_range:
        train_x, train_label, test_x, test_label = gather_train_test(pat, win, stride, freq_bins, percen_train)
        model = train_mlp(train_x, train_label, bins=freq_bins, hid_layer_nodes=30, epoch_cnt=100, plot=False)
        test_loss, test_accuracy = model.evaluate(test_x, test_label)
        patient_label += [str(pat)]
        patient_test += [test_accuracy]

    patient_test = np.array(patient_test)
    average = np.average(patient_test)
    """
    patient_test = np.concatenate((patient_test, np.array([average])))
    patient_label += ["Average"]
    patient_label = np.array(patient_label)
    plotter.bar(patient_label, patient_test)
    ax = plotter.ylim((0, 0.5))
    plotter.ylabel('Testing Accuracy')
    plotter.title('Average Testing Accuracies For All Subjects')
    plotter.show()
    """
    print("Average validation accuracy across patients: %.4f" % average)
    print("Window = %.2f, Stride = %.2f, Num_bins = %d" % (win, stride, num_bin))
    return average


if __name__ == '__main__':
    # Do not change:
    ############################
    percentage_training = 0.8
    percentage_validation = 0
    percentage_testing = 0.2
    ############################

    # Hyper Parameters
    ############################
    # set window length to be 1 second
    max_window_range = 3
    min_window_len = .1
    # set stride to be .5 seconds
    max_stride_range = 1.5
    min_stride_len = .1
    # frequency bins
    max_num_bins_range = 150
    min_num_bins = 10

    av_array = []
    label = []

    #    rng = np.random.default_rng()
    #    for i in range(10):
    #        win = rng.random() * max_window_range + min_window_len
    #        stride = rng.random() * max_stride_range + min_stride_len
    #        num_bin = int(rng.random() * max_num_bins_range + min_num_bins)
    #        print("\n#############################")
    #        print("Testing: Window=%.3f,Stride=%.3f,Bins=%d" % (win, stride, num_bin))
    #        print("#############################\n")
    #        av_array += [play_hyperparam(win, stride, num_bin, percentage_training)]
    #        label += ["W=%.1f\nS=%.1f\nB=%d" % (win, stride, num_bin)]
    #play_hyperparam(win=.5, stride=.3, num_bin=100, percen_train=percentage_training)
    #plotter.show()
    sub_label = []
    window_array = []
    stride_array = []
    acc_array = []
    for window in range(1, 6):
        actual_window = window * .6
        window_array += [actual_window]
        accuracy_array = []
        stride_array_row = []
        accuracy = []
        for stride in range(1, 6):
            actual_stride = stride * .1 * actual_window
            stride_array_row += [1 - stride * .1]

            average_acc = play_hyperparam(win=actual_window, stride=actual_stride, num_bin=100,
                                          percen_train=percentage_training)
            accuracy += [average_acc]

        stride_array += [stride_array_row]
        acc_array += [accuracy]
    fig, axes = plotter.subplots()
    axes.set_title('Stride Overlap vs. Accuracy (Window [seconds])')
    axes.set_xlabel('Stride Overlap (% Window)')
    axes.set_ylabel('Test Accuracy')
    legend_array = []
    for i, row in enumerate(stride_array):
        axes.plot(row, acc_array[i])
        legend_array += ['Window = %.3fsec' % window_array[i]]
    axes.legend(legend_array, loc='right')
    plotter.show()

    """
    patient_test = np.array(patient_test)
    average = np.average(patient_test)
    patient_test = np.concatenate((patient_test, np.array([average])))
    sub_label += ["Average"]
    sub_label = np.array(sub_label)
    plotter.bar(sub_label, patient_test)
    ax = plotter.ylim((0.4, 1))
    plotter.ylabel('Testing Accuracy')
    plotter.title('Average Testing Accuracies For All Subjects')
    plotter.show()
    """
