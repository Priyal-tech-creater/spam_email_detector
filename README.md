# Spam Email Detector

I made this project to learn how machine learning can be used to detect spam emails. It trains two models on a real dataset and compares which one does better.

Both models got around 97% accuracy which I was pretty happy with.

---
##  Features

- Text cleaning: lowercasing, tokenisation, stopword & punctuation removal
- TF-IDF vectorisation
- Two classifiers: **Multinomial Naive Bayes** and **Logistic Regression**
- Accuracy scores + confusion matrices for both models
- Single-email prediction helper

## How it works

The script takes email text, cleans it up (lowercase, removes stopwords and punctuation), converts it to TF-IDF vectors and then trains a Naive Bayes and Logistic Regression model on it. At the end it also tests a sample spam email to see what the models predict.

## Dataset

SMS Spam Collection dataset from Kaggle — https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

Download it and put `spam.csv` in the same folder as the script.

## How to run

Install dependencies:
```
pip install -r requirements.txt
```

Then just run:
```
python spam_detector.py
```

## Results

```
accuracy score of naive bayes 0.9713
accuracy score for logistic regression 0.9754
['spam']
['spam']
```

Logistic Regression came out slightly better but both were pretty close.

## Tech used

- Python
- scikit-learn
- nltk
- pandas
