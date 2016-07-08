import random
import pickle

import numpy as np
import nltk.classify.util
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression

from functional import seq

from qanta.guesser.util.functions import relu
from qanta.util.constants import N_GUESSES, DEEP_DAN_CLASSIFIER_TARGET


def compute_recall_accuracy():
    d = 300
    val_qs = pickle.load(open('data/deep/devtest', 'rb'))
    (W, b, W2, b2, W3, b3, L) = pickle.load(open('data/deep/params', 'rb'))
    classifier = pickle.load(open('data/deep/classifier', 'rb'))
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
            prev_sum += sum(L[:, sent], axis=1).reshape((d, 1))
            if len(history) == 0:
                av = np.zeros((d, 1))
            else:
                av = prev_sum / len(history)

            # apply non-linearity
            p = relu(W.dot(av) + b)
            p2 = relu(W2.dot(p) + b2)
            p3 = relu(W3.dot(p2) + b3)

            curr_feats = {}
            for dim, val in np.ndenumerate(p3):
                curr_feats['__' + str(dim)] = val

            if sent_position + 1 == len(qs):
                p_dist = classifier.prob_classify(curr_feats)
                accuracy += int(p_dist.max() == ans)
                correct = int(seq(p_dist.samples())
                              .map(lambda s: (p_dist.prob(s), s))
                              .sorted(reverse=True)
                              .take(N_GUESSES)
                              .exists(lambda s: ans == s[1]))
                recall += correct
                if not correct:
                    wrong.append((qs, ans))
            sent_position += 1
        total += 1

    return recall / total, accuracy / total, total, wrong


# trains a classifier, saves it to disk, and evaluates on heldout data
def evaluate(train_qs, test_qs, params, d):

    data = [train_qs, test_qs]
    (W, b, W2, b2, W3, b3, L) = params

    train_feats = []
    test_feats = []

    for tt, split in enumerate(data):
        for qs, ans in split:
            prev_sum = np.zeros((d, 1))
            history = []

            for dist in qs:

                sent = qs[dist]

                # input is average of all nouns in sentence
                # av = average(L[:, sent], axis=1).reshape((d, 1))
                history += sent
                prev_sum += sum(L[:, sent], axis=1).reshape((d, 1))
                if len(history) == 0:
                    av = np.zeros((d, 1))
                else:
                    av = prev_sum / len(history)

                # apply non-linearity
                p = relu(W.dot(av) + b)
                p2 = relu(W2.dot(p) + b2)
                p3 = relu(W3.dot(p2) + b3)

                curr_feats = {}
                for dim, val in np.ndenumerate(p3):
                    curr_feats['__' + str(dim)] = val

                if tt == 0:
                    train_feats.append((curr_feats, ans[0]))

                else:
                    test_feats.append((curr_feats, ans[0]))

    print('total training instances:', len(train_feats))
    print('total testing instances:', len(test_feats))
    random.shuffle(train_feats)

    # can modify this classifier / do grid search on regularization parameter using sklearn
    classifier = SklearnClassifier(LogisticRegression(C=10))
    classifier.train(train_feats)

    print('accuracy train:', nltk.classify.util.accuracy(classifier, train_feats))
    print('accuracy test:', nltk.classify.util.accuracy(classifier, test_feats))
    print('')

    print('dumping classifier')
    pickle.dump(classifier, open(DEEP_DAN_CLASSIFIER_TARGET, 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)
