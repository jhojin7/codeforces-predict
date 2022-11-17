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

from sklearn.model_selection import GridSearchCV

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
xgb_clf = Pipeline([
    ("tfidf",TfidfVectorizer(max_features=config.MAX_FEATURES)),
    ("xgb",xgb.XGBClassifier(random_state=SEED)),
])

# logistic
logistic = Pipeline([
    ("tfidf",TfidfVectorizer(max_features=config.MAX_FEATURES)),
    ("logistic",OneVsRestClassifier(LogisticRegression(random_state=SEED))),
])

def run_all():
    for name,clf in [
        ("naivebayes",tfidf_nb),
        ("xgboost",xgb_clf),
        ("logistic",logistic),
        ]:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        # print(y_pred, accuracy_score(y_test, y_pred))
        print(name,accuracy_score(y_test,y_pred))
# run_all()

xgb_param_grid = {
    "xgb__booster":["gblinear","gbtree"],
    "xgb__max_depth":[0,1,5,10,15],
    "xgb__predictor":["cpu_predictor"],
}

xgb_clf_best = GridSearchCV(xgb_clf,xgb_param_grid,n_jobs=-1,scoring="accuracy",verbose=1)
xgb_clf_best.fit(X_train,y_train)
print(xgb_clf_best.best_estimator_)
print(xgb_clf_best.best_params_)
print(xgb_clf_best.best_score_)
