

# EEG Hand Movement Decoding

The main goal of this research project was to classify hand movements from raw EEG (electroencephalogram) signals, using
machine learning and signal processing algorithms. The general idea used was to extract spectral features from the raw
EEG signals (power spectral density) and use these features to train a convolutional neural network for classification.


## File Hierarchy

* MATLAB: _MATLAB code to read the data files._
* auxiliary:
   * plotting: _functions to visualize raw signals and features._
      * plot_features.py
      * RawPSD_class.py
   * unit_tests: _unit test functions._
      * average_PSD_test.py
      * example_generation_test.py
      * PCA_on_PSD_test.py
      * power_spectral_density_test.py
      * RawPSD_class_test.py
      * spectrogram_test.py
* documentation: _documentation of algorithms implemented._
   * Feature Calculation Algorithms.pdf
   * Research Report.pdf
* models:
