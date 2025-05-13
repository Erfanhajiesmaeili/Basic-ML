# Breast Cancer Detection using Classification Models

This project focuses on predicting whether a breast tumor is malignant or benign using various machine learning classification algorithms. The dataset used is the Breast Cancer Wisconsin (Diagnostic) dataset, which is available through `sklearn.datasets`.

## Project Overview

The primary goal of this project is to build, train, and evaluate several classification models to accurately diagnose breast cancer. The workflow includes:
1.  Loading and understanding the dataset.
2.  Preprocessing the data, which involves splitting it into training and testing sets and applying feature scaling.
3.  Training multiple classification models on the prepared data.
4.  Evaluating and comparing the performance of these models based on key metrics such as accuracy, precision, and recall.

## Dataset

The dataset used is `load_breast_cancer` from `sklearn.datasets`.
* **Dataset Name:** Breast Cancer Wisconsin (Diagnostic) Dataset
* **Number of Instances:** 569
* **Number of Attributes:** 30 numeric, predictive attributes. These features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass and describe characteristics of the cell nuclei present in the image.
* **Class Labels:**
    * Malignant
    * Benign
* **Class Distribution:** 212 Malignant, 357 Benign

## Methodology

### 1. Data Loading and Exploration
The dataset is loaded using `from sklearn.datasets import load_breast_cancer`. Initial exploration involves examining the dataset's description, features, and target variables.

### 2. Preprocessing
* **Train-Test Split:** The dataset is divided into a training set (80%) and a testing set (20%) using `train_test_split` from `sklearn.model_selection`.
* **Feature Scaling:** `MinMaxScaler` from `sklearn.preprocessing` is used to scale the feature values to a range between 0 and 1. This normalization step is crucial for algorithms sensitive to feature magnitudes.

### 3. Classification Models
A variety of classification algorithms were implemented and evaluated:
* **Gaussian Naive Bayes (GNB):** Implemented using `GaussianNB` from `sklearn.naive_bayes`.
* **K-Nearest Neighbors (KNN):** Implemented using `KNeighborsClassifier` from `sklearn.neighbors` (with `n_neighbors=8`).
* **Decision Tree Classifier:** Implemented using `DecisionTreeClassifier` from `sklearn.tree` (with `max_depth=256`, `min_samples_split=2`, `criterion='gini'`).
* **Random Forest Classifier:** Implemented using `RandomForestClassifier` from `sklearn.ensemble` (with `n_estimators=1000`, `max_depth=48`, `min_samples_split=2`).
* **Support Vector Machine (SVM):** Implemented using `SVC` from `sklearn.svm` (with `kernel='poly'`).
* **Logistic Regression:** Implemented using `LogisticRegression` from `sklearn.linear_model`.
* **Artificial Neural Network (ANN) / Multi-layer Perceptron (MLP):** Implemented using `MLPClassifier` from `sklearn.neural_network` (with `hidden_layer_sizes=512`, `max_iter=1000`).

### 4. Evaluation
A custom function `calculate_metrics` was defined to compute and report the following performance metrics for both training and testing phases:
* **Accuracy:** The proportion of correct predictions among the total number of cases.
* **Precision:** The ability of the classifier not to label as positive a sample that is negative (for the positive class, typically 'Malignant').
* **Recall (Sensitivity):** The ability of the classifier to find all the positive samples.

The performance of these models on the test set (accuracy, precision, and recall) are then visualized and compared using bar charts.

## Results

The models showed varying performance levels. Based on the notebook, Logistic Regression and K-Nearest Neighbors (with the specified parameters) generally performed well on the test set, achieving high accuracy, precision, and recall. The exact metrics for each model can be found in the notebook's output cells. Visual comparisons of these metrics are provided through bar plots at the end of the notebook.

## How to Run

1.  Ensure you have Python installed.
2.  Install the required libraries:
    ```bash
    pip install scikit-learn matplotlib jupyter
    ```
3.  Open the Jupyter Notebook `Breast Cancer Detection using classification models.ipynb`.
4.  Run the cells sequentially to load the data, preprocess it, train the models, and see their evaluations and comparisons.

## Files in the Repository

* `Breast Cancer Detection using classification models.ipynb`: The Jupyter Notebook containing all the Python code, data analysis, model implementations, and results.
* `README.md`: This file, providing an overview of the project.

## Conclusion

This project successfully demonstrates the application of several common machine learning classification algorithms for breast cancer detection. The evaluation highlights that different models can achieve high performance, and the choice of model might depend on specific requirements such as interpretability, training time, or the relative importance of precision versus recall for the medical diagnosis task.