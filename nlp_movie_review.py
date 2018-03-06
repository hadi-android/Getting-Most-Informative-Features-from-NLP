# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 15:16:06 2018

@author: svchadi
"""
# source: 
# https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a

from sklearn.datasets import fetch_20newsgroups
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

#text_clf = Pipeline([('vectorizer', TfidfVectorizer()),('classifier', MultinomialNB())])
#text_clf = Pipeline([('vectorizer', CountVectorizer()),('tfidf', TfidfTransformer()),('classifier', MultinomialNB())])
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
text_clf = Pipeline([('vectorizer', TfidfVectorizer(stop_words=stop_words)),('classifier', MultinomialNB())])
text_clf = text_clf.fit(twenty_train.data, twenty_train.target)
import numpy as np
twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
predicted = text_clf.predict(twenty_test.data)
print(np.mean(predicted == twenty_test.target))

import show_most_informative_features
print(show_most_informative_features.show_most_informative_features(text_clf))

# now use logistic regression as the classifer model
from sklearn.linear_model import LogisticRegression
text_clf = Pipeline([('vectorizer', TfidfVectorizer(stop_words=stop_words)),('classifier', LogisticRegression())])
text_clf = text_clf.fit(twenty_train.data, twenty_train.target)
predicted = text_clf.predict(twenty_test.data)
print(np.mean(predicted == twenty_test.target))

print(show_most_informative_features.show_most_informative_features(text_clf))
