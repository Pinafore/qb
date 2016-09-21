import re
import os
from qanta.util.environment import QB_QUESTION_DB
from collections import Counter
import pickle

import nltk.classify.util
from nltk.util import ngrams
from unidecode import unidecode
from sklearn.linear_model import LogisticRegression
from nltk.classify.scikitlearn import SklearnClassifier

from qanta import logging
from qanta.util.io import safe_open
from qanta.util.qdb import QuestionDatabase
from qanta.util.constants import CLASSIFIER_PICKLE_PATH


log = logging.get(__name__)


alphanum = re.compile('[\W_]+')
merged = {'Biology': 'Science',
          'Physics': 'Science',
          'Chemistry': 'Science',
          'Mathematics': 'Science',
          'Earth Science': 'Science',
          'Astronomy': 'Science',
          'Social Science': 'Social Studies',
          'Geography': 'Social Studies'}


def write_bigrams(bigrams, output):
    with safe_open(output, 'wb') as f:
        pickle.dump(bigrams, f, pickle.HIGHEST_PROTOCOL)


def classify_text(classifier, text, all_bigrams):
    feats = {}
    total = alphanum.sub(' ', unidecode(text.lower()))
    total = total.split()
    bgs = set(ngrams(total, 2))
    for bg in bgs.intersection(all_bigrams):
        feats[bg] = 1.0
    for word in total:
        feats[word] = 1.0
    return classifier.prob_classify(feats)


def compute_frequent_bigrams(thresh, qbdb):
    # TODO: don't use most frequent bigrams, look by class via feature selection
    # or at least remove stopwords

    bcount = Counter()
    all_questions = qbdb.questions_with_pages()
    for page in all_questions:
        for qq in all_questions[page]:
            if qq.fold == 'train':
                for ss, ww, tt in qq.partials():
                    total = ' '.join(tt).strip()
                    total = alphanum.sub(' ', unidecode(total.lower()))
                    total = total.split()
                    bgs = list(map(str, ngrams(total, 2)))
                    for bg in bgs:
                        bcount[bg] += 1

    return set([k for k, v in bcount.most_common(thresh)])


def train_classifier(bgset, questions, class_type, limit=-1):
    all_questions = questions.questions_with_pages()
    c = Counter()
    train = []
    for page in all_questions:
        for qq in all_questions[page]:
            if qq.fold == 'train':
                label = getattr(qq, class_type, "").split(":")[0].lower()
                if not label:
                    continue
                c[label] += 1

                for ss, ww, tt in qq.partials():
                    feats = {}
                    total = ' '.join(tt).strip()
                    total = alphanum.sub(' ', unidecode(total.lower()))
                    total = total.split()

                    # add unigrams
                    for word in total:
                        feats[word] = 1.0

                    # add bigrams
                    currbg = set(ngrams(total, 2))
                    inter = currbg.intersection(bgset)
                    for elem in inter:
                        feats[elem] = 1.0

                    train.append((feats, label))
            if 0 < limit < len(train):
                break

    log.info('{}: {}'.format(class_type, c))
    log.info('{}: {}'.format(class_type, len(train)))
    log.info("{} out: training classifier".format(class_type))
    classifier = SklearnClassifier(LogisticRegression(C=10))
    classifier.train(train)
    with safe_open(CLASSIFIER_PICKLE_PATH.format(class_type), 'wb') as f:
        pickle.dump(classifier, f)
    log.info('{}: accuracy@1 train: {}'.format(
        class_type, nltk.classify.util.accuracy(classifier, train)))
    return classifier


def evaluate(bgset, questions, class_type, top=2):
    classifier_file = CLASSIFIER_PICKLE_PATH.format(class_type)
    classifier = pickle.load(open(classifier_file, 'rb'))

    all_questions = questions.questions_with_pages()
    page_num = 0
    c = Counter()
    dev = []
    for page in all_questions:
        page_num += 1
        for qq in all_questions[page]:
            if qq.fold == 'dev':
                label = getattr(qq, class_type, '').split(":")[0].lower()
                if not label:
                    continue
                c[label] += 1

                for ss, ww, tt in qq.partials():
                    feats = {}
                    total = ' '.join(tt).strip()
                    total = alphanum.sub(' ', unidecode(total.lower()))

                    # add unigrams
                    for word in total.split():
                        feats[word] = 1.0

                    # add bigrams
                    currbg = set(map(str, ngrams(total, 2)))
                    inter = currbg.intersection(bgset)
                    for elem in inter:
                        feats[elem] = 1.0

                    dev.append((feats, label))
    log.info('{}: {}'.format(class_type, c))
    log.info('{}: {}'.format(class_type, len(dev)))
    log.info('{}: accuracy@1 dev: {}'.format(
        class_type, nltk.classify.util.accuracy(classifier, dev)))
    probs = classifier.prob_classify_many([f for f, a in dev])

    corr = 0.

    for index, prob in enumerate(probs):
        c = Counter()
        for k, v in prob._prob_dict.items():
            c[k] = v
        topn = set([k for k, v in c.most_common(top)])
        if dev[index][1] in topn:
            corr += 1

    log.info('{}: top@{} accuracy: {}'.format(class_type, top, corr / len(probs)))


def build_classifier(class_type, bigram_thresh=1000):
    questions = QuestionDatabase(QB_QUESTION_DB)
    bigram_filename = CLASSIFIER_BIGRAMS_PATH.format(class_type)
    if os.path.exists(bigram_filename):
        bgset = pickle.load(open(bigram_filename, 'rb'))
        print("Using previous bigrams")
    else:
        print("computing bigrams...")
        bgset = compute_frequent_bigrams(bigram_thresh, questions)
        write_bigrams(bgset, bigram_filename)

    train_classifier(bgset, questions, class_type)
    evaluate(bgset, questions, class_type)
