# Credit-Card-Fraud-Detection-using-logistic-regression
This project detects fraudulent credit card transactions using machine learning classification algorithms. The goal is to identify suspicious transactions and reduce fraud.
# Dataset
The dataset is too large to upload to GitHub.
You can download it from Kaggle: Credit Card Fraud Detection
Instructions:
Download the dataset creditcard.csv.
Place it in the project folder temporarily to run the code.
The dataset has the following columns:
Time – seconds since first transaction
V1 to V28 – anonymized features
Amount – transaction amount
Class – target variable (0 = genuine, 1 = fraud)

# Libraries Used
pandas – for data handling
numpy – for numerical operations
matplotlib / seaborn – for visualization
scikit-learn – for machine learning models and metrics

# Steps / Code
Load dataset & check first rows.
Check class distribution (fraud vs genuine).
Split dataset into train and test sets.
Scale features using StandardScaler.
Train Logistic Regression model.
Predict test set and evaluate using:
Confusion Matrix
Classification Report

# Confusion Matrix:
[[85290   10]
 [   45   98]]
 
# Conclusion
Logistic Regression can help detect fraudulent transactions.
Evaluation metrics like precision, recall, and F1-score are important because the dataset is imbalanced.
Visualizations help understand both data distribution and model performance.
