

## models

This folder contains files for end-to-end models for feature extraction and classification..


### Folder Structure

* **classifiers:** _classification models._
  * **CNN:** _convolutional neural network implementation._
     * conv_neural_net.py: _class to build, train, and evaluate a convolutional neural network._
     * evaluate_CNN.py: _functions to train/evaluate a CNN across subjects and plot training/validation/test accuracies._
     * hyperparam_search.py: _tunes the hyperparameters of a CNN across subjects._
     * run_CNN.py: _trains and evaluates a CNN for multiple subjects._
  * example_generation.py
* **data_reading:** _Python code (calls MATLAB code) to read data files._
  * data_reader.py:
* **feature_calculation:** _feature calculation algorithms._
  * average_PSD.py:
  * feature_algorithms.py:
  * PCA_on_PSD.py:
  * power_spectral_density.py:
  * spectrogram.py:

