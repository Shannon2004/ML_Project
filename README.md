# Machine Learning Assignment

This repository contains the implementation of a machine learning assignment divided into two parts. The datasets used in this project are `train.csv` for training and `test.csv` for testing. The workflow includes preprocessing, exploratory data analysis, and applying a variety of machine learning models.

## Part 1: Preprocessing, EDA and Models
The initial stage involves performing exploratory data analysis (EDA) and preprocessing on the training and test datasets. This step is implemented in the following file:

- **`PreProcessing_And_EDA.ipynb`**
  - Analyzes the datasets to identify patterns, missing values, and outliers.
  - Applies preprocessing steps such as handling missing data, feature engineering, and scaling.
  - Outputs the processed datasets as `Processed_train.csv` and `Processed_test.csv`, which are used in subsequent modeling steps.
    
 ### Decision Tree, KNN, Random Forest, Gradient Boosting, AdaBoost, XGBoost, and Naive Bayes
 In the first part, seven models are implemented:
  - Decision Tree
  - K-Nearest Neighbors (KNN)
  - Random Forest
  - Gradient Boosting
  - AdaBoost
  - XGBoost
  - Naive Bayes    
   
Implementation details can be found in the respective files:
- **`RandomForest_AdaBoost_GradientBoosting_DecisionTrees_KNearestNeighbours.ipynb`**
  - Implements Decision Tree, KNN, Random Forest, Gradient Boosting, and AdaBoost.
- **`XGBOOST_Naive_Bayes.ipynb`**
  - Implements XGBoost and Naive Bayes.

## Part 2: Models
### Logistic Regression, SVM, and Neural Networks
In the second part, three additional models are implemented:
- Logistic Regression
- Support Vector Machines (SVM)
- Neural Networks:
  - Using PyTorch
  - Using TensorFlow

Implementation details can be found in the following files:
- **`LR_SVM_NeuralNetworkUsingPyTorch.ipynb`**
  - Implements Logistic Regression, SVM, and Neural Networks using PyTorch.
- **`NeuralNetworkUsingTensorFlow.ipynb`**
  - Implements Neural Networks using TensorFlow.


## License
This project is licensed under the MIT License. See the LICENSE file for details.
