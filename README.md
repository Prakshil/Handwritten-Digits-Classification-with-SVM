# Handwritten Digits Classification with SVM

This project demonstrates how to classify handwritten digits (0-9) using a Support Vector Machine (SVM) algorithm. It builds a complete machine learning pipeline comprising data loading, exploratory visualization, model training, evaluation, hyperparameter tuning, and model persistence.

## Overview

The project uses the `sklearn` digits dataset, which contains 1,797 samples of 8x8 pixel grayscale images corresponding to digits from 0 to 9. The classifier is built using `scikit-learn` and employs an RBF (Radial Basis Function) kernel for highly accurate predictions. 

## Features
- **Data Visualization**: Renders image arrays into visual grayscale formats to develop an intuition for the dataset.
- **Data Preprocessing**: Splits data continuously and normalizes the features utilizing `StandardScaler` to boost model performance stability.
- **SVM Classification**: Employs an SV Classifier within a `scikit-learn` `Pipeline` object for seamless data transformations and inferencing.
- **Model Evaluation**: Generates metrics including Accuracy, Classification Report (Precision, Recall, F1), and a Confusion Matrix to thoroughly evaluate predictive strength.
- **Hyperparameter Tuning**: Utilizes `GridSearchCV` to iterate over ranges of `C` and `gamma` parameters, identifying the best metrics for the RBF SVM.
- **Model Serialization**: Saves the final trained model utilizing `pickle` (`digit_classifier.pkl`) for rapid deployment or inference outside the notebook.

## Project Structure

- `Digit Predictor.ipynb`: The main Jupyter Notebook containing the end-to-end Python code for the model pipeline, evaluation, and visualizations.
- `digit_classifier.pkl`: The serialized (`pickle`) output model, generated after fitting the pipeline (created upon running the notebook).
- `README.md`: This descriptive file.

## Requirements

Ensure you have the following packages installed:
- `numpy`
- `matplotlib`
- `scikit-learn`
- `joblib` (though `pickle` is being used for final serialization)

You can install them via pip:
```bash
pip install numpy matplotlib scikit-learn
```

## Usage

1. Open `Digit Predictor.ipynb` in your preferred Jupyter environment (e.g., Jupyter Notebook, JupyterLab, or VS Code).
2. Execute the cells sequentially.
3. The final notebook block will export the best-fit model into `digit_classifier.pkl` within the same directory.
4. You can then load `digit_classifier.pkl` into any custom python script utilizing `pickle` to predict new 8x8 image representations.