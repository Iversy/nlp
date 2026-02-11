
from pathlib import Path
from loader import load 
from cleaner import clean
from broing import _tokenise, get_tokens_word, get_tokens_sent
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

    data = load(path)
    data = clean(data)
    print("-"*40)
    print(data.info())
    print("-"*40)
    print(data.describe())

    tokens_word = get_tokens_word(data)
    print("-"*40)
    
    print(tokens_word.info())
    print("-"*40)
    print(tokens_word.describe())
    tokens_word = tokens_word.apply(len)
    print("-"*40)
    
    print(tokens_word.info())
    print("-"*40)
    print(tokens_word.describe())
    tokens_sent = get_tokens_sent(data)
    print("-"*40)
    
    print(tokens_sent.info())
    print("-"*40)
    print(tokens_sent.describe())
    tokens_sent = tokens_sent.apply(len)

    print("-"*40)
    
    print(tokens_sent.info())
    print("-"*40)
    print(tokens_sent.describe())


if __name__ == "__main__":
    main()