""" 
Spam Email Detector
====================
Classifies emails as spam or ham using:
  - Multinomial Naive Bayes
  - Logistic Regression
with TF-IDF feature extraction and NLTK text preprocessing.
"""

import pandas as pd
import sklearn
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

df = pd.read_csv("spam.csv", encoding="latin1")
X = df["v2"]
Y = df["v1"]

eng = set(stopwords.words('english'))

#Text processing

def clean_text(text):
    text = text.lower()
    c = word_tokenize(text)
    new_df = []
    for word in c:
        if word not in eng and word not in string.punctuation:
            new_df.append(word)
    return " ".join(new_df)

X = X.apply(clean_text)

#Training and Testing

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

tdf = TfidfVectorizer()
new_X_train = tdf.fit_transform(X_train)
new_X_test = tdf.transform(X_test)

# naive bayes
model1 = MultinomialNB()
model1.fit(new_X_train, Y_train)
y_predict = model1.predict(new_X_test)

# logistic regression
model2 = LogisticRegression(max_iter=1000)
model2.fit(new_X_train, Y_train)
y_predict2 = model2.predict(new_X_test)

# test email
my_email = "Congratulations! You have won $1,000,000 lottery prize. Click here to claim now"
clean = clean_text(my_email)
email = tdf.transform([clean])
result = model1.predict(email)
new_result = model2.predict(email)

result1 = accuracy_score(Y_test, y_predict)
result2 = accuracy_score(Y_test, y_predict2)

print("accuracy score of naive bayes", result1)
print("accuracy score for logistic regression", result2)
print(result)
print(new_result)
