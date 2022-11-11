import pickle
import pandas as pd
import numpy as np
import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.preprocessing import MultiLabelBinarizer
# try:
#     stop_words = set(stopwords.words("english"))
# except LookupError:
#     nltk.download("stopwords",config.NLTK_DIR)
# nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def remove_mathjax(text):
    mathjax = "(\$\$\$(.*?)\$\$\$)"
    shortword = r"\W*\b\w{1,2}\b"
    return "".join(re.sub(mathjax," ",text))

def remove_stopwords(text):
    return " ".join(wordpunct_tokenize(text))


def remove_shorts_stopwords(text):
    result = []
    # for word in tokenizer.tokenize(regex_pipeline):
    for word in text.split():
        if len(word)>=3 and word not in stop_words:
            result.append(word)
    return " ".join(result)

def stemming(text):
    stemmer = SnowballStemmer("english")
    words = set()
    for word in text.split():
        stem = stemmer.stem(word)
        words.add(stem)
    return " ".join(words)

# text preprocessing as function
def preprocess_text(text):
    """ preprocessing text 
    return processed text
    - text = text to process
    """
    text = remove_mathjax(text)
    text = remove_stopwords(text)
    text = remove_shorts_stopwords(text)
    text = stemming(text)
    return text

def preprocess_df(df):
    """ returns df """
    df["problem_statement"] = \
        df["problem_statement"].apply(lambda x:preprocess_text(str(x)))
    return df

def binarize_y(y):
    """ Binarize y """
    mlb = MultiLabelBinarizer()
    y_binarized = pd.DataFrame(mlb.fit_transform(y),
        columns=mlb.classes_, index=y.index)
    return y_binarized