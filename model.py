import pickle
import pandas as pd
import numpy as np

import config
import preprocess

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, classification_report, hamming_loss)
import xgboost as xgb

SEED = config.SEED
train = pd.read_pickle(config.TRAIN)
train = preprocess.preprocess_df(train)
X_train, y_train = train["problem_statement"], train["tags"]
y_train = preprocess.binarize_y(y_train)

test = pd.read_pickle(config.TEST)
test = preprocess.preprocess_df(test)
X_test, y_test = test["problem_statement"], test["tags"]
y_test = preprocess.binarize_y(y_test)

# naive_bays
nb_clf = Pipeline([
    ("tfidf",TfidfVectorizer(max_features=config.MAX_FEATURES)),
    ("multi_nb",OneVsRestClassifier(MultinomialNB())),
])

# xgboost
xgb_clf = Pipeline([
    ("tfidf",TfidfVectorizer(max_features=config.MAX_FEATURES)),
    ("xgb",xgb.XGBClassifier(random_state=SEED)),
])

# linear
lin_clf = Pipeline([
    ("tfidf",TfidfVectorizer(max_features=config.MAX_FEATURES)),
    ("linear",OneVsRestClassifier(LinearRegression())),
])

# logistic
log_clf = Pipeline([
    ("tfidf",TfidfVectorizer(max_features=config.MAX_FEATURES)),
    ("logistic",OneVsRestClassifier(LogisticRegression(random_state=SEED))),
])

# xgb_param_grid = {
#     "xgb__booster":["gblinear","gbtree"],
#     "xgb__max_depth":[0,1,5,10,15],
#     "xgb__predictor":["cpu_predictor"],
# }
# xgb_clf_best = GridSearchCV(
#     xgb_clf,xgb_param_grid,
#     n_jobs=-1,scoring="accuracy",verbose=1)
# xgb_clf_best.fit(X_train,y_train)
# print(xgb_clf_best.best_estimator_)
# print(xgb_clf_best.best_params_)
# print(xgb_clf_best.best_score_)

ALL_MODELS = [lin_clf, log_clf, nb_clf, xgb_clf]