# This file contains code to plot raw signal values and various features.


import numpy as np
import matplotlib.pyplot as plotter
from models.classifiers import example_generation
from models.feature_calculation import power_spectral_density as power
from models.feature_calculation import feature_algorithms
from models.feature_calculation import average_PSD
from plotting import RawPSD_class


print("\n")

# NOT TO BE MODIFIED:
# path to data files:
path_to_data_file = "../MATLAB/biosig/Data_txt/"
# sampling frequency:
sample_freq = 250

# subject number:
subject_num = 7

# examples to plot:
examples = np.array([49, 249, 449])
num_plot_examples = examples.shape[0]
# channels to plot:
channels = np.array([0, 2])
num_plot_channels = channels.shape[0]
# names of all channels:
channel_names = np.array(['C1', 'C2', 'C3'])

# HYPERPARAMETERS:
# for creating more training examples:
window_size_example = 3.5
stride_size_example = 0.1
# for average-bin PSD features:
max_freq = 30
num_bins = 10
# for PCA on PSD features:
num_pcs = num_bins
matrix_type = 2
small_param = 0.0001
# for spectrogram generation:
window_size_PSD = 1.0
stride_size_PSD = 0.05
max_freq_spectro = 30
num_bins_spectro = 40


# get data and generate examples:
X, Y = example_generation.generate_examples(subject_num, path_to_data_file, window_size_example, stride_size_example,
                                            sample_freq)
# display dimensions of raw data:
print("Size of raw data set: ", end="")
print(X.shape)
print("")
num_examples = X.shape[0]
num_channels = X.shape[1]
num_samples = X.shape[2]

# estimate power spectral density:
Rxx, freq, PSD = power.estimate_psd(X, sample_freq)
# create Raw_PSD object:
raw_psd_object = RawPSD_class.RawPSD(num_examples, num_channels, num_samples, sample_freq, X, Rxx, freq, PSD)

# display selected raw signals:
raw_psd_object.plot_raw_signal(examples, channels, channel_names)
# display selected power spectral densities:
raw_psd_object.plot_PSD(examples, channels, channel_names)


# construct frequency bins for PSD average calculation:
bin_width = max_freq / num_bins
bins = np.zeros((num_bins, 2))
for i in range(num_bins):
    bins[i, 0] = i * bin_width
    bins[i, 1] = (i + 1) * bin_width
# calculate average PSD values in selected frequency bins:
PSD_avg = feature_algorithms.average_PSD_algorithm(X, bins, sample_freq)
# plot bar graph:
average_PSD.plot_average_PSD(PSD_avg, bins, examples, channels, channel_names)

# calculate average PSD values in frequency bins with PCA:
PSD_avg_PCA = feature_algorithms.PCA_on_PSD_algorithm(X, sample_freq, max_freq, num_bins, num_pcs=num_bins,
                                                      matrix_type=matrix_type, small_param=small_param)
# plot bar graph:
average_PSD.plot_average_PSD(PSD_avg_PCA, bins, examples, channels, channel_names)


# generate spectrogram features without PCA:
PCA = 0
X_spectro = feature_algorithms.spectrogram_algorithm(X, window_size_PSD, stride_size_PSD, sample_freq, max_freq_spectro,
                                                     num_bins_spectro, PCA=PCA, num_pcs=num_pcs,
                                                     matrix_type=matrix_type, small_param=small_param)
# display dimensions of spectrogram images:
print("Size of spectrogram images without PCA: ", end="")
print(X_spectro.shape)
# create and format subplot:
fig, axes = plotter.subplots(nrows=num_plot_examples, ncols=num_plot_channels)
# reshape axes to handle shape error if either dimension = 1
axes = np.reshape(axes, (num_plot_examples, num_plot_channels))
plotter.subplots_adjust(hspace=1)

for i, axis in enumerate(axes.flat):
    # get indices of example and channel:
    example_ind = i // num_plot_channels
    channel_ind = i % num_plot_channels
    axis.imshow(np.transpose(X_spectro[examples[example_ind], channels[channel_ind]]), aspect='equal',
                origin='lower')
    axis.set_title('Spectrogram of Example {0}, {1}'.format(examples[example_ind] + 1,
                                                            channel_names[channels[channel_ind]]))
plotter.tight_layout(True)

# generate spectrogram features with PCA:
PCA = 1
X_spectro = feature_algorithms.spectrogram_algorithm(X, window_size_PSD, stride_size_PSD, sample_freq, max_freq_spectro,
                                                     num_bins_spectro, PCA=PCA, num_pcs=num_bins_spectro,
                                                     matrix_type=matrix_type, small_param=small_param)
# display dimensions of spectrogram images:
print("Size of spectrogram images with PCA: ", end="")
print(X_spectro.shape)
# create and format subplot:
fig, axes = plotter.subplots(nrows=num_plot_examples, ncols=num_plot_channels)
# reshape axes to handle shape error if either dimension = 1
axes = np.reshape(axes, (num_plot_examples, num_plot_channels))
plotter.subplots_adjust(hspace=1)

for i, axis in enumerate(axes.flat):
    # get indices of example and channel:
    example_ind = i // num_plot_channels
    channel_ind = i % num_plot_channels
    axis.imshow(np.transpose(X_spectro[examples[example_ind], channels[channel_ind]]), aspect='equal',
                origin='lower')
    axis.set_title('Spectrogram of Example {0}, {1}'.format(examples[example_ind] + 1,
                                                            channel_names[channels[channel_ind]]))
plotter.tight_layout(True)


# display plots:
plotter.show()
