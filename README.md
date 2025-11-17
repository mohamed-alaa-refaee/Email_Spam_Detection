# 📧 Spam Email Classification Project

A comprehensive machine learning pipeline for classifying SMS or email messages as spam or ham (non-spam) using traditional NLP techniques and supervised learning algorithms.
This project demonstrates the full workflow from data preprocessing to model selection, optimization, handling class imbalance, and evaluation inside a single Jupyter Notebook.

## 📂 Dataset

The project uses the SMS Spam Collection Dataset provided by the UCI Machine Learning Repository, containing labeled text messages categorized as ham or spam.

## 🚀 Project Pipeline

The notebook includes the following major steps:

### 1. Data Loading & Exploration

- Importing the dataset

- Handling duplicates

- Label encoding (spam = 1, ham = 0)

### 2. Exploratory Data Analysis (EDA)

- Distribution of classes

- Count plots, pie charts

- Checking for class imbalance

### 3. Text Preprocessing (NLTK)

- Tokenization

- Lowercasing

- Removing punctuation & stopwords

- Extracting statistics (sentence/word/character counts)

### 4. Feature Engineering

- Vectorization using CountVectorizer

- Train/test split

### 5. Baseline Model Benchmarking

- Using LazyPredict to quickly test multiple classical ML models

### 6. Hyperparameter Optimization

- Tuning Bernoulli Naive Bayes via GridSearchCV

### 7. Handling Class Imbalance

- Applying RandomUnderSampler (from imbalanced-learn)

### 8. Model Evaluation

- Metrics used:

- Confusion Matrix

- Accuracy, Precision, Recall, F1-Score

- ROC Curve & AUC

- Classification Report

### 9. Ensemble Modeling

- A VotingClassifier combining:

- Decision Tree

- BernoulliNB

- Random Forest

## 🏆 Best Model Performance

After applying RandomUnderSampler, Bernoulli Naive Bayes achieved:

Metric	Score
- Accuracy	~98.65%
- Precision	~97.20%
- Recall	~93.29%
- F1-Score	~95.21%
## 🛠️ Requirements

- Make sure the following packages are installed:

>pip install pandas numpy matplotlib seaborn scikit-learn nltk lazypredict xgboost imbalanced-learn

## ▶️ How to Run

- Clone the repository

- git clone <your-repo-link>


- Install the dependencies

- Launch Jupyter Notebook

- jupyter notebook


- Open the notebook and run all cells to reproduce the results.

## 📜 License

This project is licensed under the MIT License.
