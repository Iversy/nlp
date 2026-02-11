import pandas as pd
from pathlib import Path
import docx
from pypdf import PdfReader
import re 
## Часть 1

def load_csv(path: Path, sep: str = ",") -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["text"])["text"]
    return df

def load_txt(path: Path, sep: str = "SEP") -> pd.DataFrame:
    text = path.read_text()
    parts = text.split(sep)
    parts = [p.strip() for p in parts if p]
    return pd.Series(parts, dtype="string").reset_index(drop=True)

def load_docx(path: Path, sep: str = "SEP") -> pd.Series:
    doc = docx.Document(path)
    text = ''.join(paragraph.text for paragraph in doc.paragraphs)
    parts = [p.strip() for p in text.split(sep)]
    return pd.Series(parts, dtype="string").reset_index(drop=True)

def load_pdf(path: Path, sep="SEP") -> pd.Series:
    reader = PdfReader(path)
    text = "".join(page.extract_text() for page in reader.pages)
    parts = [p.strip() for p in text.split(sep)]
    return pd.Series(parts, dtype="string").reset_index(drop=True)



def load(path: Path) -> pd.DataFrame | pd.Series:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    match path.suffix:
        case ".csv":
            return load_csv(path)
        case ".txt":
            return load_txt(path)
        case ".docx":
            return load_docx(path)
        case ".pdf":
            return load_pdf(path)
        case _:
            raise ValueError
        