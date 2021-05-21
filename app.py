from flask import Flask, render_template, url_for, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import string
import csv
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.metrics import classification_report

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from string import punctuation


def preprocess(corpus):
    """Process tweet function.
    Input:
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet

    """
    words=[]
    stemmer = PorterStemmer()
    for i in corpus:

        word_tokens = [word.lower() for word in word_tokenize(i)]
        t=""
        for word in word_tokens:
            if(not word in set(stopwords.words('english')) and  not word in string.punctuation and word.isalpha()):
                t+=stemmer.stem(word.lower())+" "
        words.append(t)
    # tokenize tweets

    return words


# from sklearn.externals import joblib
app = Flask(__name__)


# Machine Learning code goes here
@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    df = pd.read_csv("data/Youtube01-Psy.csv")
    df_data = df[['CONTENT', 'CLASS']]
    # Features and Labels
    df_x = df_data['CONTENT']
    df_y = df_data.CLASS
    # Extract the features with countVectorizer
    corpus = preprocess(df_x)
    cv = CountVectorizer()
    X = cv.fit_transform(corpus)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size=0.33, random_state=42)
    # Navie Bayes
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)
    print(clf.score(X_test, y_test))
    if request.method == 'POST':
        comment = request.form['comment']
        data = [comment]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5555, debug=True)