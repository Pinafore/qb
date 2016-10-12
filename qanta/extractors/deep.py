import warnings
import pickle
import numpy as np
from string import ascii_lowercase, punctuation

from functional import seq
from unidecode import unidecode

from qanta.extractors.abstract import FeatureExtractor
from qanta.util.constants import PAREN_EXPRESSION, STOP_WORDS, N_GUESSES


valid_strings = set(ascii_lowercase) | set(str(x) for x in range(10)) | {' '}


def relu(x):
    return x * (x > 0)


def normalize(text):
    text = unidecode(text).lower().translate(str.maketrans(punctuation, ' ' * len(punctuation)))
    text = PAREN_EXPRESSION.sub("", text)
    text = " ".join(x for x in text.split() if x not in STOP_WORDS)
    return ''.join(x for x in text if x in valid_strings)


class DeepExtractor(FeatureExtractor):
    def __init__(self, classifier, params, vocab, ners, page_dict):
        super(DeepExtractor, self).__init__()
        self.classifier = pickle.load(open(classifier, 'rb'), encoding='latin1')
        self.params = pickle.load(open(params, 'rb'), encoding='latin1')
        self.d = self.params[-1].shape[0]
        self.vocab, self.vdict = pickle.load(open(vocab, 'rb'), encoding='latin1')
        self.ners = pickle.load(open(ners, 'rb'), encoding='latin1')
        self.page_dict = page_dict
        self.name = 'deep'

    def set_metadata(self, answer, category, qnum, sent, token, guesses, fold):
        pass

    # return a vector representation of the question
    def compute_rep(self, text):
        text = ' '.join(text)
        curr_feats = self.compute_features(text)
        return curr_feats, text

    # return a distribution over answers for the given question
    def compute_probs(self, text):
        curr_feats = self.compute_features(text)
        return self.classifier.predict_proba(curr_feats)[0]

    def compute_features(self, text: str):
        W, b, W2, b2, W3, b3, L = self.params

        # generate word vector lookups given normalized input text
        text = normalize(text)
        for ner in self.ners:
            text = text.replace(ner, ner.replace(' ', '_'))

        inds = []
        for w in text.split():
            if w in self.vdict:
                inds.append(self.vdict[w])

        if len(inds) > 0:
            # compute vector representation for question text
            av = np.average(L[:, inds], axis=1).reshape((self.d, 1))
            p = relu(np.dot(W, av) + b)
            p2 = relu(np.dot(W2, p) + b2)
            p3 = relu(np.dot(W3, p2) + b3)

        else:
            p3 = np.zeros((self.d, 1))

        return p3.ravel().reshape(1, -1)

    # return top n guesses for a given question
    def text_guess(self, text):
        text = ' '.join(text)
        preds = self.compute_probs(text)
        class_labels = self.classifier.classes_
        guesses = seq(preds).zip(class_labels).sorted(reverse=True).take(N_GUESSES)

        res = {}
        for p, word in guesses:
            res[self.page_dict[self.vocab[word]]] = p

        return res

    def score_one_guess(self, title, text):
        if isinstance(text, list):
            text = ' '.join(text)

        preds = self.compute_probs(text)

        # return -1 if classifier doesn't recognize the given guess
        if title in self.vdict:
            guess_ind = self.vdict[title]
            val = preds[guess_ind]
        else:
            val = -1

        return val

    def score_guesses(self, guesses, text):
        if isinstance(text, list):
            warnings.warn(
                "use of list or str input text in deep.score_guesses is deprecated",
                DeprecationWarning
            )
            text = ' '.join(text)

        labels = self.classifier.classes_
        predictions = self.compute_probs(text)
        lookup = dict(zip(labels, predictions))
        for guess in guesses:
            if guess in lookup:
                yield self.vw_from_score(lookup[guess]), guess
            else:
                yield self.vw_from_score(-1), guess

    def vw_from_score(self, val):
        res = "|%s" % self.name
        if val == -1:
            res += " deepfound:0 deepscore:0.0"
        else:
            res += " deepfound:1 deepscore:%f" % val

        return res
