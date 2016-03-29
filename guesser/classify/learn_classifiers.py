from numpy import *
import nltk.classify.util
from util.math_util import *
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression
from scipy import stats
from collections import Counter
import cPickle, csv, random, argparse

def check_accuracy_at_1():
    d=300
    val_qs = cPickle.load(open('data/deep/dev', 'rb'))
    (W, b, W2, b2, W3, b3, L) = cPickle.load(open('data/deep/params', 'rb'))

    test_feats = []
    for qs, ans in test_qs:

        prev_qs = zeros((d, 1))
        prev_sum = zeros((d, 1))
        count = 0.
        history = []

        for dist in qs:

            sent = qs[dist]

            # input is average of all nouns in sentence
            # av = average(L[:, sent], axis=1).reshape((d, 1))
            history += sent
            prev_sum += sum(L[:, sent], axis=1).reshape((d, 1))
            if len(history) == 0:
                av = zeros((d, 1))
            else:
                av = prev_sum / len(history)

            # apply non-linearity
            p = relu(W.dot(av) + b)
            p2 = relu(W2.dot(p) + b2)
            p3 = relu(W3.dot(p2) + b3)

            curr_feats = {}
            for dim, val in ndenumerate(p3):
                curr_feats['__' + str(dim)] = val

            test_feats.append( (curr_feats, ans[0]) )

    print 'total testing instances:', len(test_feats)

    # can modify this classifier / do grid search on regularization parameter using sklearn
    classifier = cPickle.load(open('data/deep/classifier', 'rb'))
    print 'accuracy test:', nltk.classify.util.accuracy(classifier, test_feats)

# trains a classifier, saves it to disk, and evaluates on heldout data
def evaluate(train_qs, test_qs, params, d):

    data = [train_qs, test_qs]
    (W, b, W2, b2, W3, b3, L) = params

    train_feats = []
    test_feats = []

    for tt, split in enumerate(data):

        for qs, ans in split:

            prev_qs = zeros((d, 1))
            prev_sum = zeros((d, 1))
            count = 0.
            history = []

            for dist in qs:

                sent = qs[dist]

                # input is average of all nouns in sentence
                # av = average(L[:, sent], axis=1).reshape((d, 1))
                history += sent
                prev_sum += sum(L[:, sent], axis=1).reshape((d, 1))
                if len(history) == 0:
                    av = zeros((d, 1))
                else:
                    av = prev_sum / len(history)

                # apply non-linearity
                p = relu(W.dot(av) + b)
                p2 = relu(W2.dot(p) + b2)
                p3 = relu(W3.dot(p2) + b3)

                curr_feats = {}
                for dim, val in ndenumerate(p3):
                    curr_feats['__' + str(dim)] = val

                if tt == 0:
                    train_feats.append( (curr_feats, ans[0]) )

                else:
                    test_feats.append( (curr_feats, ans[0]) )

    print 'total training instances:', len(train_feats)
    print 'total testing instances:', len(test_feats)
    random.shuffle(train_feats)

    # can modify this classifier / do grid search on regularization parameter using sklearn
    classifier = SklearnClassifier(LogisticRegression(C=10))
    classifier.train(train_feats)

    print 'accuracy train:', nltk.classify.util.accuracy(classifier, train_feats)
    print 'accuracy test:', nltk.classify.util.accuracy(classifier, test_feats)
    print ''

    print 'dumping classifier'
    cPickle.dump(classifier, open('data/deep/classifier', 'wb'),
                 protocol=cPickle.HIGHEST_PROTOCOL)
