import pickle
import re
from functools import lru_cache
from typing import Tuple, List

from qanta.util.constants import NERS_PATH
from sklearn.feature_extraction.text import CountVectorizer


FOR_TEN_POINTS = ["for 10 points, ", "for 10 points--", "for ten points, ", "for 10 points ",
                  "for ten points ", "ftp,", "ftp"]


def clean_question(question: str):
    q = question.strip().lower()

    # remove pronunciation guides and other formatting extras
    q = q.replace(' (*) ', ' ')
    q = q.replace('\n', '')
    q = q.replace('mt. ', 'mt ')
    q = q.replace(', for 10 points, ', ' ')
    q = q.replace(', for ten points, ', ' ')
    q = q.replace('--for 10 points--', ' ')
    q = q.replace(', ftp, ', ' ')
    q = q.replace('{', '')
    q = q.replace('}', '')
    q = q.replace('~', '')
    q = q.replace('(*)', '')
    q = q.replace('*', '')
    q = re.sub(r'\[.*?\]', '', q)
    q = re.sub(r'\(.*?\)', '', q)

    for phrase in FOR_TEN_POINTS:
        q = q.replace(phrase, ' ')

    # remove punctuation
    q = re.sub(r"\p{P}+", " ", q)
    return q


@lru_cache(maxsize=None)
def load_ners():
    with open(NERS_PATH, 'rb') as f:
        return pickle.load(f)


def replace_named_entities(question):
    for ner in load_ners():
        question = question.replace(ner, ner.replace(' ', '_'))
    return question


def format_guess(guess):
    return guess.strip().lower().replace(' ', '_').replace(':', '').replace('|', '')


def preprocess_dataset(data: Tuple[List[List[str]], List[str]]):
    x_data = []
    y_data = []
    for q, ans in zip(data[0], data[1]):
        for run in q:
            x_data.append(run)
            y_data.append(ans)

    def cv_preprocess(text):
        return replace_named_entities(clean_question(text))

    cv = CountVectorizer(preprocessor=cv_preprocess)
    return cv.fit_transform(x_data, y_data)


class Preprocessor:
    def __init__(self):
        # map vocab to word embedding lookup index
        self.vocab = []
        self.vdict = {}

    def preprocess_dataset(self):
        pass

    def preprocess_question(self, q: str):
        q = clean_question(q)
        q = replace_named_entities(q)

        words = self.convert_to_indices(q.strip())
        return words

    def convert_to_indices(self, text):
        words = []
        for w in text.split():
            if w not in self.vdict:
                self.vocab.append(w)
                self.vdict[w] = len(self.vocab) - 1
            words.append(self.vdict[w])
        return words
