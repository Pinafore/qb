import pickle
import re
from functools import lru_cache
from typing import Tuple, List, Set
import string

from qanta.util.constants import NERS_PATH
from qanta import logging
from nltk import word_tokenize
import numpy as np


log = logging.get(__name__)


def clean_question(question: str):
    """
    Remove pronunciation guides and other formatting extras
    :param question:
    :return:
    """
    patterns = {
        '\n',
        ', for 10 points,',
        ', for ten points,',
        '--for 10 points--',
        'for 10 points, ',
        'for 10 points--',
        'for ten points, ',
        'for 10 points ',
        'for ten points ',
        ', ftp,'
        'ftp,',
        'ftp'
    }

    patterns |= set(string.punctuation)
    regex_pattern = '|'.join([re.escape(p) for p in patterns])
    regex_pattern += r'|\[.*?\]|\(.*?\)'

    return re.sub(regex_pattern, '', question.strip().lower())


@lru_cache(maxsize=None)
def load_ners():
    log.info('Loading ners file...')
    with open(NERS_PATH, 'rb') as f:
        return pickle.load(f)


def replace_named_entities(question: str):
    for ner in load_ners():
        question = question.replace(ner, ner.replace(' ', '_'))
    return question


def format_guess(guess):
    return guess.strip().lower().replace(' ', '_').replace(':', '').replace('|', '')


def preprocess_dataset(data: Tuple[List[List[str]], List[str]]):
    for i in range(len(data[1])):
        data[1][i] = format_guess(data[1][i])
    classes = set(data[1])
    class_to_i = {}
    i_to_class = []
    for i, ans_class in enumerate(classes):
        class_to_i[ans_class] = i
        i_to_class.append(ans_class)

    x_data = []
    y_data = []
    vocab = set()

    for q, ans in zip(data[0], data[1]):
        for run in q:
            q_text = word_tokenize(clean_question(run))
            for w in q_text:
                vocab.add(w)
            x_data.append(q_text)
            y_data.append(class_to_i[ans])

    return x_data, y_data, vocab, class_to_i, i_to_class


GLOVE_WE = 'data/external/deep/glove.6B.300d.txt'


def create_embeddings(vocab: Set[str]):
    embeddings = []
    embedding_lookup = {}
    with open(GLOVE_WE) as f:
        i = 0
        for l in f:
            splits = l.split()
            word = splits[0]
            if word in vocab:
                emb = [float(n) for n in splits[1:]]
                embeddings.append(emb)
                embedding_lookup[word] = i
                i += 1
        return np.array(embeddings), embedding_lookup
