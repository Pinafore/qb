import re
from typing import Tuple, List
import string

from qanta import logging
from nltk import word_tokenize
from sklearn.cross_validation import train_test_split

from qanta.datasets.abstract import TrainingData

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


def tokenize_question(text: str) -> List[str]:
    return word_tokenize(clean_question(text))


def format_guess(guess):
    return guess.strip().lower().replace(' ', '_').replace(':', '').replace('|', '')


def preprocess_dataset(data: TrainingData, train_size=.9,
                       vocab=None, class_to_i=None, i_to_class=None):
    for i in range(len(data[1])):
        data[1][i] = format_guess(data[1][i])
    classes = set(data[1])

    if class_to_i is None or i_to_class is None:
        class_to_i = {}
        i_to_class = []

    for ans_class in classes:
        if ans_class not in class_to_i:
            class_to_i[ans_class] = len(class_to_i)
            i_to_class.append(ans_class)

    x_train = []
    y_train = []
    properties_train = []
    x_test = []
    y_test = []
    properties_test = []
    if vocab is None:
        vocab = set()

    question_runs_with_answer = list(zip(data[0], data[1], (data[2] if data[2] else [None for _ in data[0]])))
    if train_size != 1:
        train, test = train_test_split(question_runs_with_answer, train_size=train_size)
    else:
        train = question_runs_with_answer
        test = []

    for q, ans, prop in train:
        for sentence in q:
            q_text = tokenize_question(sentence)
            if len(q_text) > 0:
                for w in q_text:
                    vocab.add(w)
                x_train.append(q_text)
                y_train.append(class_to_i[ans])
                properties_train.append(prop)

    for q, ans, prop in test:
        for sentence in q:
            q_text = tokenize_question(sentence)
            x_test.append(q_text)
            y_test.append(class_to_i[ans])
            properties_test.append(prop)

    return (x_train, y_train, properties_train,
            x_test, y_test, properties_test,
            vocab, class_to_i, i_to_class)
