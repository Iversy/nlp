from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
# from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import pymorphy3
from additional import additional_stopwords

stemmer = SnowballStemmer('russian')

# russian_stopwords = stopwords.words('russian')
all_stopwords = set( additional_stopwords)
morph = pymorphy3.MorphAnalyzer()


def flatten(x):
    for item in x:
        if isinstance(item, (list, tuple)):
            yield from flatten(item)
        else:
            yield item

def _token_word(text):
    return word_tokenize(text, language="russian")
def _token_sent(text):
    return sent_tokenize(text, language="russian")


def get_tokens_sent(series: pd.Series) -> pd.Series:
    series_parts = flatten([_token_sent(row) for row in series])
    return pd.Series(series_parts, dtype="string").reset_index(drop=True)


def get_tokens_word(series: pd.Series) -> pd.Series:
    series_parts = flatten([_token_word(row) for row in series])
    return pd.Series(series_parts, dtype="string").reset_index(drop=True)

def stem_text(tokens):
    return pd.Series([stemmer.stem(word) for word in tokens], dtype='string').reset_index(drop=True)


def lemmatize_text(tokens):
    lemmas = []
    for word in tokens:
        # Анализ слова и выбор наиболее вероятной формы
        parsed = morph.parse(word)[0]
        lemmas.append(parsed.normal_form)
    return pd.Series(lemmas, dtype="string").reset_index(drop=True)


def delete_stop(series: pd.Series) -> pd.Series:
    result = []
    for token in series:
        if not token in all_stopwords: 
            result.append(token)
    
    return pd.Series(result, dtype="string").reset_index(drop=True)

def _tokenise(series: pd.Series) -> pd.Series:
    result = get_tokens_word(series)
    result = delete_stop(result)
    result = stem_text(result)
    result = lemmatize_text(result)
    return result