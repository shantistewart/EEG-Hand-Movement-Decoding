# This file contains 2 functions:
#   1) Feature calculation algorithm 1: average PSD values in selected frequency bins.
#   2) Feature calculation algorithm 2: Principal Component Analysis on PSD values.


from feature_calculation import power_spectral_density as power
from feature_calculation import average_PSD
from feature_calculation import PCA_on_PSD


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
def average_PSD_algorithm(X, sample_freq, bins):
    # estimate power spectral density:
    Rxx, freq, PSD = power.estimate_psd(X, sample_freq)

    # calculate average PSD values in bins:
    PSD_avg = average_PSD.calc_average_PSD(PSD, bins, sample_freq)

    return PSD_avg


# Function description: performs channel-specific PCA on log-normalized PSD values
# Inputs:
#   X = 3D array of signal values for multiple channels for multiple examples
#       size: (num_examples, num_channels, num_samples)
#   sample_freq = sampling frequency
#   num_pcs = number of principal components (eigenvectors) to project onto
#       validity: num_pcs <= num_freq
#   small_param = a small number to ensure that log(0) does not occur
#   matrix_type = input to select which type of statistical matrix to calculate:
#       if matrix type == 1: unnormalized autocorrelation matrices are calculated
#       if matrix type == 2: unnormalized autocovariance matrices are calculated
#       else: Pearson autocovariance matrices are calculated
# Outputs:
#   project_weights = 3D array of projection weights onto principal components (eigenvectors)
#       size: (num_examples, num_channels, num_pcs)
def PCA_on_PSD_algorithm(X, sample_freq, num_pcs, small_param, matrix_type):
    # estimate power spectral density:
    Rxx, freq, PSD = power.estimate_psd(X, sample_freq)

    # log-normalize PSD values across examples:
    PSD_norm = PCA_on_PSD.log_normalize(PSD, small_param)

    # calculate selected statistical matrices of PSD values:
    if matrix_type == 1:
        stat_matrices = PCA_on_PSD.unnorm_correlation(PSD)
    elif matrix_type == 2:
        stat_matrices = PCA_on_PSD.unnorm_covariance(PSD)
    else:
        stat_matrices = PCA_on_PSD.pearson_covariance(PSD)

    # calculate eigenvectors of PSD statistical matrices:
    eig_vects = PCA_on_PSD.calc_eig_vects(stat_matrices)

    # calculates projection weights of PSD values onto principal components (eigenvectors):
    project_weights = PCA_on_PSD.project_onto_pcs(PSD, eig_vects, num_pcs)

    return project_weights
