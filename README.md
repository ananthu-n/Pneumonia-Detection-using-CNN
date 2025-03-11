# Pneumonia-Detection-using-CNN
**Project Overview**

This project uses Convolutional Neural Networks (CNNs) to classify chest X-ray images as either Pneumonia or Normal. The model is trained on a dataset of pediatric X-rays from the Guangzhou Women and Children’s Medical Center. The goal is to develop an automated and accurate diagnostic system to assist medical professionals in detecting pneumonia from X-ray images.

**Dataset Description**
- The dataset is sourced from the Guangzhou Women and Children’s Medical Center.
- It contains 5,863 X-ray images categorized into two classes:
    -  Pneumonia
    -  Normal
- The dataset is divided into training, validation, and test sets.
- Images are in JPEG format and were screened by medical professionals.
  
**Model Architecture**
  
  The deep learning model is built using Convolutional Neural Networks (CNN) in TensorFlow and Keras. The architecture includes:

- Convolutional layers with ReLU activation
- Max pooling layers for feature extraction
- Fully connected dense layers
- Dropout layers to prevent overfitting
- Softmax activation for binary classification

**Training Process**

- Data Preprocessing
  
    - Image resizing and normalization
    - Data augmentation to improve generalization

- Model Training

    - CNN model trained using categorical cross-entropy loss
    - Adam optimizer with a learning rate of 0.001
    - Early stopping to avoid overfitting

- Evaluation

    - Model tested on unseen data
    - Metrics: Accuracy, Precision, Recall, F1-score
 
Results & Performance Metrics
- Training Accuracy: 95.64%
- Validation Accuracy: 87.50%
- Loss and accuracy plotted for better analysis
- Confusion matrix used for evaluating misclassifications

Installation & Usage Guide

Prerequisites

Ensure you have the following installed:
  
  - Python 3.x
  - TensorFlow
  - Keras
  - NumPy
  - OpenCV
  - Matplotlib
