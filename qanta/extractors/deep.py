import warnings
import numpy as np
from string import ascii_lowercase, punctuation

from functional import seq

from qanta.extractors.abstract import AbstractFeatureExtractor
from qanta.util.environment import QB_QUESTION_DB
from qanta.util.constants import PAREN_EXPRESSION, STOP_WORDS, N_GUESSES
from qanta.datasets.quiz_bowl import QuestionDatabase
from qanta.preprocess import format_guess


valid_strings = set(ascii_lowercase) | set(str(x) for x in range(10)) | {' '}


def relu(x):
    return x * (x > 0)


def normalize(text):
    text = text.lower().translate(str.maketrans(punctuation, ' ' * len(punctuation)))
    text = PAREN_EXPRESSION.sub("", text)
    text = " ".join(x for x in text.split() if x not in STOP_WORDS)
    return ''.join(x for x in text if x in valid_strings)


class DeepExtractor(AbstractFeatureExtractor):
    def __init__(self):
        super(DeepExtractor, self).__init__()
        question_db = QuestionDatabase(QB_QUESTION_DB)
        page_dict = {}
        for page in question_db.get_all_pages():
            page_dict[page.lower().replace(' ', '_')] = page
        self.page_dict = page_dict

    @property
    def name(self):
        return 'deep'

    def compute_rep(self, text):
        """
        return a vector representation of the question
        :param text:
        :return:
        """
        text = ' '.join(text)
        curr_feats = self.compute_features(text)
        return curr_feats, text

    def compute_probs(self, text):
        """
        return a distribution over answers for the given question
        :param text:
        :return:
        """
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

    def text_guess(self, text):
        """
        return top n guesses for a given question
        :param text:
        :return:
        """
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
            formatted_guess = format_guess(guess)
            guess_id = self.vdict[formatted_guess]
            if guess_id in lookup:
                yield self.vw_from_score(lookup[guess_id])
            else:
                yield self.vw_from_score(-1)

    def vw_from_score(self, val):
        res = "|%s" % self.name
        if val == -1:
            res += " deepfound:0 deepscore:0.0"
        else:
            res += " deepfound:1 deepscore:%f" % val

        return res
