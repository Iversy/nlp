from pathlib import Path
from loader import load 
from cleaner import clean
from broing import _tokenise
import argparse
import nltk

parser = argparse.ArgumentParser(description='NLP')
parser.add_argument('-p', '--path', type=str, required=False, help='path to file')
args = parser.parse_args()


def main():
    if args.path is None:
        raise BaseException("No path provided")
    path = args.path
    path = Path(path)


    nltk.download('punkt')
    nltk.download('punkt_tab')
    # nltk.download('stopwords')

    serega = load(path)
    print(serega)
    sergei = clean(serega)
    print(sergei)
    sergei_sergeevich = _tokenise(sergei)
    print(sergei_sergeevich)

if __name__ == "__main__":
    main()