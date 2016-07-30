import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from functional import seq

from qanta import logging
from qanta.guesser.util.functions import relu
from qanta.util.constants import (N_GUESSES, DEEP_DAN_CLASSIFIER_TARGET, DEEP_DAN_PARAMS_TARGET,
                                  DEEP_DEV_TARGET)
from qanta.util.io import safe_open

log = logging.get(__name__)


def compute_recall_accuracy():
    d = 300
    with open(DEEP_DEV_TARGET, 'rb') as f:
        val_qs = pickle.load(f)
    with open(DEEP_DAN_PARAMS_TARGET, 'rb') as f:
        (W, b, W2, b2, W3, b3, L) = pickle.load(f)
    with open( DEEP_DAN_CLASSIFIER_TARGET, 'rb') as f:
        classifier = pickle.load(f)
        class_labels = classifier.classes_
    recall = 0
    accuracy = 0
    total = 0
    wrong = []
    for qs, ans in val_qs:
        ans = ans[0]
        prev_sum = np.zeros((d, 1))
        history = []
        sent_position = 0
        for dist in qs:

            sent = qs[dist]

            # input is average of all nouns in sentence
            # av = average(L[:, sent], axis=1).reshape((d, 1))
            history += sent
            prev_sum += np.sum(L[:, sent], axis=1).reshape((d, 1))
            if len(history) == 0:
                av = np.zeros((d, 1))
            else:
                av = prev_sum / len(history)

            # apply non-linearity
            p = relu(W.dot(av) + b)
            p2 = relu(W2.dot(p) + b2)
            p3 = relu(W3.dot(p2) + b3)

            curr_feats = p3.ravel().reshape(1, -1)

            if sent_position + 1 == len(qs):
                p_dist = classifier.predict_proba(curr_feats)
                correct = seq(zip(p_dist[0], class_labels))\
                    .sorted(reverse=True)\
                    .take(N_GUESSES)\
                    .exists(lambda s: ans == s[1])
                accuracy += int(class_labels[p_dist.argmax()] == ans)
                recall += correct
                if not correct:
                    wrong.append((qs, ans))
            sent_position += 1
        total += 1

    return recall / total, accuracy / total, total, wrong


def compute_vectors(train_qs, test_qs, params, d):
    data = [train_qs, test_qs]
    (W, b, W2, b2, W3, b3, L) = params

    train_vector = []
    test_vector = []

    for tt, split in enumerate(data):
        for qs, ans in split:
            prev_sum = np.zeros((d, 1))
            history = []

            for dist in qs:

                sent = qs[dist]

                # input is average of all nouns in sentence
                # av = average(L[:, sent], axis=1).reshape((d, 1))
                history += sent
                prev_sum += np.sum(L[:, sent], axis=1).reshape((d, 1))
                if len(history) == 0:
                    av = np.zeros((d, 1))
                else:
                    av = prev_sum / len(history)

                # apply non-linearity
                p = relu(W.dot(av) + b)
                p2 = relu(W2.dot(p) + b2)
                p3 = relu(W3.dot(p2) + b3)

                if tt == 0:
                    train_vector.append((p3.ravel(), ans[0]))

                else:
                    test_vector.append((p3.ravel(), ans[0]))
    return train_vector, test_vector


# trains a classifier, saves it to disk, and evaluates on heldout data
def evaluate(train_vector, test_vector):
    log.info('total training instances: {0}'.format(len(train_vector[0])))
    log.info('total testing instances: {0}'.format(len(test_vector[0])))

    classifier = OneVsRestClassifier(LogisticRegression(C=10), n_jobs=-1)
    classifier.fit(train_vector[0], train_vector[1])

    with safe_open(DEEP_DAN_CLASSIFIER_TARGET, 'wb') as f:
        pickle.dump(classifier, f, protocol=pickle.HIGHEST_PROTOCOL)

    train_accuracy = classifier.score(X=train_vector[0], y=train_vector[1])
    test_accuracy = classifier.score(X=test_vector[0], y=test_vector[1])
    log.info('accuracy train: {0}'.format(train_accuracy))
    log.info('accuracy test: {0}'.format(test_accuracy))
