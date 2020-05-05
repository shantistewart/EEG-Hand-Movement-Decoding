# This file contains 2 functions:
#   1) Feature calculation algorithm 1: average PSD values in selected frequency bins.
#   2) Feature calculation algorithm 2: Principal Component Analysis on PSD values.


from feature_calculation import power_spectral_density as power
from feature_calculation import average_PSD
from feature_calculation import PCA_on_PSD as PCA


# Function description: calculate average power spectral density values in selected frequency bins.
# Inputs:
#   X = 3D array of signal values for multiple channels for multiple examples
#       size: (num_examples, num_channels, num_samples)
#   sample_freq = sampling frequency
#   bins = frequency bins to calculate average PSD values for
#       size: (num_bins, 2)
# Outputs:
#   PSD_bins = 3D array of average PSD values in selected frequency bins for multiple channels for multiple examples
#       size: (num_examples, num_channels, num_bins)
