

## models

This folder contains files to implement end-to-end models for feature extraction and classification.


### Folder Structure

* **classifiers:** _classification models._
  * **CNN:** _convolutional neural network implementation._
     * conv_neural_net.py: _class to build, train, and evaluate a convolutional neural network._
     * evaluate_CNN.py: _functions to train/evaluate a CNN across subjects and plot training/validation/test accuracies._
     * hyperparam_search.py: _tunes the hyperparameters of a CNN across subjects._
     * run_CNN.py: _trains and evaluates a CNN for multiple subjects._
  * example_generation.py: _functions to generate training/validation/test examples for training and evaluation._
* **data_reading:** _Python code to read data files._
  * data_reader.py: _function to gather data from data files._
* **feature_calculation:** _feature calculation algorithms._
  * PCA_on_PSD.py: _functions to implement principal component analysis on power spectral density values._
  * average_PSD.py: _functions to calculate average power spectral density values in frequency bins and plot values._
  * feature_algorithms.py: _3 feature calculation algorithms -- frequency bin-average PSD, PCA on PSD, PSD spectrograms._
  * power_spectral_density.py: _function to estimates power spectral density._
  * spectrogram.py: _function to generate power spectral density spectrograms._

