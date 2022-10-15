import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


import config
train = pd.read_pickle("./data/train.pkl")
test = pd.read_pickle("./data/test.pkl")
X_train, y_train = train["problem_statement"], train["tags"]
X_test, y_test = test["problem_statement"], test["tags"]

print(X_train.tail())
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# multilabel binarizer
mlb = MultiLabelBinarizer()
y_train_onehot = pd.DataFrame(mlb.fit_transform(y_train),
    columns=mlb.classes_, index=y_train.index)
y_test_onehot = pd.DataFrame(mlb.fit_transform(y_test),
    columns=mlb.classes_, index=y_test.index)

print(mlb.classes_)
print(y_train_onehot.shape)
print(X_train.shape, y_train_onehot["implementation"].shape)

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
print(X_train_counts.shape)
X_test_counts = count_vect.transform(X_test)
print(X_test_counts.shape)

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()

tags = mlb.classes_
for tag in tags:
    y_train_imp = y_train_onehot[tag]
    clf.fit(X_train_counts, y_train_imp)
    y_pred = clf.predict(X_test_counts)
    print(y_pred, len(y_pred))
    # for i,n in enumerate(y_pred):
    #     if n==1: print(tags[i], end=' ')
    # print()

# print(f"{tag} accuracy:",
#     accuracy_score(y_test_onehot[tag],y_pred))