

# EEG Hand Movement Decoding

The main goal of this research project was to classify hand movements from raw EEG (electroencephalogram) signals, using
machine learning and signal processing algorithms. The general idea used was to extract spectral features from the raw
EEG signals (power spectral density) and use these features to train a convolutional neural network (CNN) for classification.



## Required Python Libraries

* TensorFlow
* NumPy
* Scikit-learn
* Matplotlib


## Repository Structure

* **MATLAB:** _MATLAB code to read data files._
* **auxiliary:** _auxiliary functionality._
   * **plotting:** _visualization of raw signals and features._
      * plot_features.py
      * RawPSD_class.py
   * **unit_tests**: _unit tests._
      * average_PSD_test.py
      * example_generation_test.py
      * PCA_on_PSD_test.py
      * power_spectral_density_test.py
      * RawPSD_class_test.py
      * spectrogram_test.py
* **documentation:** _documentation of algorithms implemented._
   * Feature Calculation Algorithms.pdf
   * Research Report.pdf
* **models:** _end-to-end models for feature extraction and classification._
   * **classifiers:** _classification models._
      * **CNN:** _convolutional neural network implementation._
         * conv_neural_net.py
         * evaluate_CNN.py
         * hyperparam_search.py
         * run_CNN.py
      * example_generation.py
   * **data_reading:** _Python code (calls MATLAB code) to read data files._
      * data_reader.py
   * **feature_calculation:** _feature calculation algorithms._
      * average_PSD.py
      * feature_algorithms.py
      * PCA_on_PSD.py
      * power_spectral_density.py
      * spectrogram.py


## Scripts (files that can be run)

* **run_CNN.py:** _trains and evaluates a CNN model for multiple subjects._
   * file path: models/classifiers/CNN/run_CNN.py
* **hyperparam_search.py:** _tunes hyperparameters of a CNN model across multiple subjects._
   * file path: models/classifiers/CNN/hyperparam_search.py
* **plot_features.py:** _plots raw EEG signals and several calculated features._
   * file path: auxiliary/plotting/plot_features.py
* unit tests:
   * **power_spectral_density_test.py:** _unit tests for functions in power_spectral_density.py._
      * file path: auxiliary/unit_tests/power_spectral_density_test.py
   * **average_PSD_test.py:** _unit tests for functions in average_PSD.py._
      * file path: auxiliary/unit_tests/average_PSD_test.py
   * **PCA_on_PSD_test.py:** _unit tests for functions in PCA_on_PSD.py._
      * file path: auxiliary/unit_tests/PCA_on_PSD_test.py
   * **spectrogram_test.py:** _unit tests for functions in spectrogram.py._
      * file path: auxiliary/unit_tests/spectrogram_test.py
   * **example_generation_test.py:** _unit tests for functions in example_generation.py._
      * file path: auxiliary/unit_tests/example_generation_test.py
   * **RawPSD_class_test.py:** _unit tests for functions in RawPSD_class.py._
      * file path: auxiliary/unit_tests/RawPSD_class_test.py


## Authors

* **Shanti Stewart** -- B.S. student in Electrical and Computer Engineering at Oregon State University.
   * [LinkedIn](https://www.linkedin.com/in/shanti-stewart-27aaaa183/)
* **Patrick McGrath** -- B.S. student in Electrical and Computer Engineering at Oregon State University.
   * [LinkedIn](https://www.linkedin.com/in/mcgrath-patrick)

