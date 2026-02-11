import pandas as pd
import re

def _clean(text: str) -> str:
    text = text.lower()
    text, _ = re.subn('[0-9]',"", text)
    text, _ = re.subn('[,.;:]', '', text)
    text, _ = re.subn(' +', ' ', text)
    return text

def clean(text_series: pd.Series) -> pd.Series:
    result = text_series.copy()
    result = result.apply(_clean)
    return result
