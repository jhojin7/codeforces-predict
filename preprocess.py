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


def remove_stopwords(text):
    return " ".join(wordpunct_tokenize(text))


def remove_shorts_stopwords(text):
    """ remove short words
    input: string
    output: string
    """
    result = []
    # for word in tokenizer.tokenize(regex_pipeline):
    for word in text.split():
        if len(word)>=3 and word not in stop_words:
            result.append(word)
    return " ".join(result)

def stemming(text):
    """ use stemmer 
    input: string
    output: string
    """
    stemmer = SnowballStemmer("english")
    words = set()
    for word in text.split():
        stem = stemmer.stem(word)
        words.add(stem)
    return " ".join(words)

# text preprocessing as function
def preprocess_text(text):
    """ preprocessing text 
    input: string
    output: string
    """
    # all lowercase
    text = text.lower()
    # regex $$$mathjax$$$
    text = re.sub(
        r"(\$\$\$(.*?)\$\$\$)","",text)
    # regex !@#$...
    text = re.sub(
        r'[@%\\*=()/~#&\+á?\xc3\xa1\-\|\.\:\;\!\-\,\_\~\$\'\"\n\]\[\>]', '',text)
    # 
    text = remove_stopwords(text)
    text = stemming(text)
    text = remove_stopwords(text)
    text = remove_shorts_stopwords(text)
    return text

def clean_tags_column(tags):
    _tags = str(tags).split(',')
    ret = []
    for tag in _tags:
        # remove rating tag  
        if '*' not in tag:
            ret.append(tag)
    return ret

def binarize_y(y):
    """ Binarize y """
    mlb = MultiLabelBinarizer()
    y_binarized = pd.DataFrame(mlb.fit_transform(y),
        columns=mlb.classes_, index=y.index)
    return y_binarized