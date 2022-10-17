import streamlit as st
import pickle
# jinja ImportError
# pip install Jinja2 --upgrade
import config
from preprocess import preprocess_text

#####
import nltk
nltk.download("stopwords")
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import wordpunct_tokenize
# from nltk.stem.snowball import SnowballStemmer
# try:
#     stop_words = set(stopwords.words("english"))
# except LookupError:
#     nltk.download("stopwords",config.NLTK_DIR)
#####

st.title("codeforces_tag_predict")
X_in = st.text_area("Enter raw text here:")

model = pickle.load(open(config.MODEL, 'rb'))
pipeline, tags = model

X_raw = preprocess_text(X_in)
# predict through pipeline
y_pred = pipeline.predict([X_raw])

def tags_as_str(y_pred):
    y_ans = []
    for i in range(len(y_pred)):
        if y_pred[i] == 1: 
            y_ans.append(tags[i])
    return y_ans
y_pred_str = tags_as_str(y_pred[0])

st.text(f"""\
- Converted Raw Input: {X_raw}
- y_pred: {y_pred[0]}
- Predicted Tags: {y_pred_str}
""")