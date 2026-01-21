# Breast Cancer Prediction with Neural Networks

This project implements a feedforward neural network in PyTorch to classify breast tumours as benign or malignant using the Breast Cancer dataset from scikit-learn. The goal is to explore training stability, generalization, and regularization techniques in neural networks through controlled experiments.

The notebook evaluates different optimization strategies and overfitting prevention techniques, and compares their impact on accuracy, precision, recall, and training dynamics.

## Features

- Feedforward neural network with fully connected layers and ReLU activations
- Binary classification with sigmoid output
- Custom implementation of:
  - Accuracy
  - Precision
  - Recall
- Multiple training strategies:
  - Stochastic Gradient Descent with early stopping
  - Mini-batch SGD
- Regularization experiments:
  - Dropout (multiple dropout rates)
  - L1 and L2 weight regularization
- Visualization of training and validation loss curves
- Comparative analysis of generalization behavior


## Model Architecture

The network consists of:
- Input layer matching the dataset feature size
- Hidden layer (32 neurons, ReLU)
- Hidden layer (16 neurons, ReLU)
- Output layer (1 neuron with sigmoid activation)

## Dataset

- **Dataset:** Breast Cancer Wisconsin Dataset  
- **Source:** scikit-learn built-in dataset  
- **Samples:** 569  
- **Task:** Binary classification (benign vs malignant)  

Dataset documentation:  
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html

The dataset is loaded directly through scikit-learn, no dataset files are included in this repository.

## Experiments

The notebook explores:
- Early stopping behavior and convergence stability
- Mini-batch training efficiency and performance
- Impact of dropout rates on overfitting
- Effect of L1 vs L2 regularization on generalization and sparsity
- Tradeoffs between precision and recall
