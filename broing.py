from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import pymorphy3

from additional import additional_stopwords

stemmer = SnowballStemmer('russian')

russian_stopwords = stopwords.words('russian')
all_stopwords = set(russian_stopwords + additional_stopwords)
morph = pymorphy3.MorphAnalyzer()


def flatten(x):
    for item in x:
        if isinstance(item, (list, tuple)):
            yield from flatten(item)
        else:
            yield item

def _token(text):
    return sent_tokenize(text, language="russian")

def get_tokens(series: pd.Series) -> pd.Series:
    series_parts = flatten([_token(row) for row in series])
    return pd.Series(series_parts, dtype="string").reset_index(drop=True)

def stem_text(tokens):
    return [stemmer.stem(word) for word in tokens]


def lemmatize_text(tokens):
    lemmas = []
    for word in tokens:
        # Анализ слова и выбор наиболее вероятной формы
        parsed = morph.parse(word)[0]
        lemmas.append(parsed.normal_form)
    return pd.Series(lemmas, dtype="string").reset_index(drop=True)


def token_clean(series: pd.Series) -> pd.Series:

    pass