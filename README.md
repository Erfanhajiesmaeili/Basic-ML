# Digit Recognition

This project focuses on recognizing handwritten digits from images using various machine learning classification algorithms. The dataset used for this project is the "Optical recognition of handwritten digits dataset" available through `sklearn.datasets`.

## Project Overview

The goal of this project is to build and evaluate different classification models capable of accurately identifying digits (0-9) from 8x8 pixel images. The process involves:
1.  Loading and understanding the dataset.
2.  Preprocessing the data, including splitting into training and testing sets and scaling the features.
3.  Applying dimensionality reduction using Principal Component Analysis (PCA).
4.  Training and evaluating several classification models:
    * Random Forest
    * Support Vector Machine (SVM)
    * Artificial Neural Network (ANN) / Multi-layer Perceptron (MLP)
    * K-Nearest Neighbors (KNN)
5.  Comparing the performance of these models based on metrics like accuracy, precision, and recall.

## Dataset

The dataset used is `load_digits` from `sklearn.datasets`.
* **Number of Instances:** 1797
* **Number of Attributes:** 64 (each instance is an 8x8 image flattened into a 64-element vector)
* **Attribute Information:** Each attribute represents a pixel intensity, with integer values in the range 0 to 16.
* **Classes:** 10 (digits 0 through 9)

## Methodology

### 1. Data Loading and Exploration
The dataset is loaded using `from sklearn.datasets import load_digits`. Basic exploration includes checking the shape of the data and visualizing some sample images using `matplotlib`.

### 2. Preprocessing
* **Train-Test Split:** The data is split into training (80%) and testing (20%) sets using `train_test_split` from `sklearn.model_selection`.
* **Feature Scaling:** `MinMaxScaler` from `sklearn.preprocessing` is used to scale the pixel values to a range between 0 and 1. This helps in improving the performance of some algorithms.

### 3. Dimensionality Reduction (PCA)
Principal Component Analysis (`PCA` from `sklearn.decomposition`) is applied to reduce the dimensionality of the data. In this project, the number of components was set to 32. This step can help in reducing computation time and potentially improving model performance by removing redundant features.

### 4. Classification Models
The following classification algorithms were implemented and evaluated:

* **Random Forest:**
    * Implemented using `RandomForestClassifier` from `sklearn.ensemble`.
    * Hyperparameters: `max_depth=128`, `n_estimators=100`.
* **Support Vector Machine (SVM):**
    * Implemented using `SVC` from `sklearn.svm`.
    * Hyperparameters: `kernel='poly'`.
* **Artificial Neural Network (ANN):**
    * Implemented using `MLPClassifier` from `sklearn.neural_network`.
    * Hyperparameters: `hidden_layer_sizes=256`, `batch_size=64`, `learning_rate='adaptive'`.
* **K-Nearest Neighbors (KNN):**
    * Implemented using `KNeighborsClassifier` from `sklearn.neighbors`.
    * Hyperparameters: `n_neighbors=4`.

### 5. Evaluation
A custom function `calculate_metrics` was defined to compute and print the following metrics for both training and testing sets:
* **Accuracy:** The proportion of correctly classified instances.
* **Precision (macro):** The ability of the classifier not to label as positive a sample that is negative. Macro-averaged precision calculates precision for each label and finds their unweighted mean.
* **Recall (macro):** The ability of the classifier to find all the positive samples. Macro-averaged recall calculates recall for each label and finds their unweighted mean.

The performance of the models (accuracy, precision, and recall on the test set) are then compared using bar charts.

## Results

The performance of each model on the test set was as follows (exact values can be found in the notebook output):

* **Random Forest:**
    * Test Accuracy: ~98.06%
    * Test Precision: ~98.00%
    * Test Recall: ~97.99%
* **SVM (poly kernel):**
    * Test Accuracy: ~99.72%
    * Test Precision: ~99.68%
    * Test Recall: ~99.74%
* **ANN (MLP):**
    * Test Accuracy: ~98.61%
    * Test Precision: ~98.69%
    * Test Recall: ~98.36%
* **KNN (n_neighbors=4):**
    * Test Accuracy: ~98.33%
    * Test Precision: ~98.43%
    * Test Recall: ~98.20%

Based on the results, the SVM with a polynomial kernel achieved the highest accuracy, precision, and recall on the test set for this specific hyperparameter configuration and data split.

## How to Run

1.  Ensure you have Python installed.
2.  Install the required libraries:
    ```bash
    pip install scikit-learn matplotlib jupyter
    ```
3.  Open the Jupyter Notebook `num.ipynb`.
4.  Run the cells sequentially to reproduce the results.

## Files in the Repository

* `num.ipynb`: The Jupyter Notebook containing all the code, analysis, and results.
* `README.md`: This file.

## Conclusion

This project demonstrates the application of various machine learning classifiers for the task of handwritten digit recognition. The SVM model showed excellent performance, highlighting its effectiveness for this type of image classification task. Further improvements could potentially be achieved by more extensive hyperparameter tuning, exploring other preprocessing techniques, or using more complex neural network architectures.