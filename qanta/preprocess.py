import re
from typing import Tuple, List
import string

from qanta import logging
from nltk import word_tokenize
from sklearn.cross_validation import train_test_split


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


def preprocess_dataset(data: Tuple[List[List[str]], List[str]], train_size=.9):
    for i in range(len(data[1])):
        data[1][i] = format_guess(data[1][i])
    classes = set(data[1])
    class_to_i = {}
    i_to_class = []
    for i, ans_class in enumerate(classes):
        class_to_i[ans_class] = i
        i_to_class.append(ans_class)

    x_train = []
    y_train = []
    x_test = []
    y_test = []
    vocab = set()

    question_runs_with_answer = list(zip(data[0], data[1]))
    train, test = train_test_split(question_runs_with_answer, train_size=train_size)

    for q, ans in train:
        for run in q:
            q_text = tokenize_question(run)
            for w in q_text:
                vocab.add(w)
            x_train.append(q_text)
            y_train.append(class_to_i[ans])

    for q, ans in test:
        for run in q:
            q_text = tokenize_question(run)
            x_test.append(q_text)
            y_test.append(class_to_i[ans])

    return x_train, y_train, x_test, y_test, vocab, class_to_i, i_to_class
