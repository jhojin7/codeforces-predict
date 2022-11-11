import pickle
import pandas as pd
import numpy as np

import config
import preprocess

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

SEED = config.SEED
train = pd.read_pickle(config.TRAIN)
train = preprocess.preprocess_df(train)
X_train, y_train = train["problem_statement"], train["tags"]
y_train = preprocess.binarize_y(y_train)

test = pd.read_pickle(config.TEST)
test = preprocess.preprocess_df(test)
X_test, y_test = test["problem_statement"], test["tags"]
y_test = preprocess.binarize_y(y_test)

# tfidf+naive_bays
tfidf_nb = Pipeline([
    ("tfidf",TfidfVectorizer(max_features=config.MAX_FEATURES)),
    ("multi_nb",OneVsRestClassifier(MultinomialNB())),
])

# tfidf + xgboost
tfidf_xgb = Pipeline([
    ("tfidf",TfidfVectorizer(max_features=config.MAX_FEATURES)),
    ("xgboost",xgb.XGBClassifier(random_state=SEED)),
])

# logistic
logistic = Pipeline([
    ("tfidf",TfidfVectorizer(max_features=config.MAX_FEATURES)),
    ("logistic",OneVsRestClassifier(LogisticRegression(random_state=SEED))),
])

for pipe in [
    tfidf_nb,
    tfidf_xgb,
    logistic,
    ]:
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    # print(y_pred, accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test,y_pred))
