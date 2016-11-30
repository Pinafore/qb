import pickle
import re
from functools import lru_cache

from qanta.util.constants import NERS_PATH


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
