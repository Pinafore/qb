import re
from typing import List
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
                       vocab=None, class_to_i=None, i_to_class=None,
                       create_runs=False, full_question=False):
    if full_question and create_runs:
        raise ValueError('The options create_runs={} and full_question={} are not compatible'.format(
            create_runs, full_question))

    for i in range(len(data[1])):
        data[1][i] = format_guess(data[1][i])
    classes = set(data[1])
    if class_to_i is None or i_to_class is None:
        class_to_i = {}
        i_to_class = []
        for i, ans_class in enumerate(classes):
            class_to_i[ans_class] = i
            i_to_class.append(ans_class)

    x_train = []
    y_train = []
    properties_train = []
    x_test = []
    y_test = []
    properties_test = []
    if vocab is None:
        vocab = set()

    question_runs_with_answer = list(zip(data[0], data[1], data[2]))
    if train_size != 1:
        train, test = train_test_split(question_runs_with_answer, train_size=train_size)
    else:
        train = question_runs_with_answer
        test = []

    for q, ans, prop in train:
        q_text = []
        for sentence in q:
            t_question = tokenize_question(sentence)
            if create_runs or full_question:
                q_text.extend(t_question)
            else:
                q_text = t_question
            if len(t_question) > 0:
                for w in t_question:
                    vocab.add(w)
                if create_runs:
                    x_train.append(list(q_text))
                elif not full_question:
                    x_train.append(q_text)

                if not full_question:
                    y_train.append(class_to_i[ans])
                    properties_train.append(prop)
        if full_question:
            x_train.append(q_text)
            y_train.append(class_to_i[ans])
            properties_train.append(prop)

    for q, ans, prop in test:
        q_text = []
        for sentence in q:
            t_question = tokenize_question(sentence)
            if create_runs or full_question:
                q_text.extend(t_question)
                if not full_question:
                    x_test.append(list(q_text))
            else:
                q_text = t_question
                x_test.append(q_text)
            if not full_question:
                y_test.append(class_to_i[ans])
                properties_test.append(prop)
        if full_question:
            x_test.append(q_text)
            y_test.append(class_to_i[ans])
            properties_test.append(prop)

    return (x_train, y_train, properties_train,
            x_test, y_test, properties_test,
            vocab, class_to_i, i_to_class)
