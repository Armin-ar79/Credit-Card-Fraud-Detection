# 💳 Credit Card Fraud Detection

A machine learning project to detect fraudulent credit card transactions using Python, Scikit-learn, and the Imbalanced-learn library.

## 🎯 Project Goal

The primary challenge of this dataset is its severe imbalance: **only 0.17%** of transactions are fraudulent. A model that always predicts "Not Fraud" would be 99.83% accurate but completely useless.

The goal was to build a model that can effectively **identify** the fraudulent transactions (Class 1) by prioritizing **High Recall**, even if it means sacrificing some precision.

## 🛠️ Tech Stack & Methodology

- **Language:** Python
- **Libraries:** Pandas, Scikit-learn, Imbalanced-learn (imblearn), Matplotlib, Seaborn
- **Model:** Logistic Regression
- **Key Technique:** **SMOTE (Synthetic Minority Over-sampling Technique)** was applied *only* to the training data to create a balanced dataset for the model to learn from, without tainting the test set.

## 📈 Results

By training the model on the balanced (SMOTE) data, we achieved excellent results on the unbalanced test set:

- **Recall (Fraud Class): 86%**
  - The model successfully identified 86% of all actual fraudulent transactions.
- **Accuracy (Overall): 98.3%**
  - While overall accuracy is high, the high recall for the minority class is the true measure of success.

This demonstrates a successful strategy for handling highly imbalanced datasets in a real-world scenario.

## 🚀 How to Run

### 1. Get the Data
Download the dataset from Kaggle and place `creditcard.csv` in the root folder:
- **Kaggle Dataset:** [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

### 2. Install Dependencies
```bash
pip install pandas scikit-learn imbalanced-learn matplotlib seaborn

### 3. Run the Scripts
First, explore the data imbalance:
py eda.py

Then, train the model using SMOTE and see the results:
py train.py
