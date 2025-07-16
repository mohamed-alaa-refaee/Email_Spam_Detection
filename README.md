# Spam Email Classification Project

This repository contains a Jupyter Notebook for classifying SMS or email messages as spam or ham (non-spam) using classic machine learning and natural language processing techniques. The notebook includes the full pipeline from data loading and cleaning to feature engineering, model training, hyperparameter tuning, resampling for class imbalance, ensemble modeling, and evaluation.

The dataset used is the SMS Spam Collection Dataset from the UCI Machine Learning Repository. It contains labeled text messages as either ham or spam.

Main steps in the notebook:

- Data loading and initial exploration
- Data cleaning (deduplication, label encoding)
- Exploratory Data Analysis (EDA) with count plots and pie charts
- Text preprocessing using NLTK (tokenization, sentence/word/character counts)
- Feature extraction with CountVectorizer
- Train/test splitting
- Model benchmarking with LazyPredict
- Hyperparameter tuning using GridSearchCV (BernoulliNB)
- Addressing class imbalance with RandomUnderSampler
- Model evaluation using confusion matrix, classification report, accuracy, precision, recall, F1-score, ROC curve, and AUC
- Ensemble modeling with VotingClassifier (combining DecisionTree, BernoulliNB, RandomForest)

Best results achieved with BernoulliNB (after RandomUnderSampler):

- Accuracy: ~98.65%
- F1-score: ~95.21%
- Precision: ~97.20%
- Recall: ~93.29%

Requirements:

- Python 3.x
- Jupyter Notebook
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- nltk
- lazypredict
- xgboost
- imbalanced-learn

Install dependencies:

pip install pandas numpy matplotlib seaborn scikit-learn nltk lazypredict xgboost imbalanced-learn

How to run:

1. Clone the repository
2. Install the requirements
3. Open the notebook in Jupyter
4. Run all cells to reproduce results

License: MIT

Acknowledgements: UCI Machine Learning Repository for the dataset, scikit-learn, NLTK, LazyPredict, XGBoost.
