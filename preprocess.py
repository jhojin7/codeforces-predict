import config

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.stem.snowball import SnowballStemmer
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords",config.NLTK_DIR)

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
    df["problem_statement"] = \
        df["problem_statement"].apply(lambda x:preprocess_text(str(x)))
    return df