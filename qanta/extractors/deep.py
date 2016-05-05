import pickle
import numpy as np
from string import ascii_lowercase, punctuation
from collections import Counter

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

    @staticmethod
    def has_guess():
        return True

    # return a vector representation of the question
    def compute_rep(self, text):
        W, b, W2, b2, W3, b3, L = self.params

        # generate word vector lookups given normalized input text
        text = normalize(' '.join(text))
        for ner in self.ners:
            text = text.replace(ner, ner.replace(' ', '_'))

        inds = []
        for w in text.split():
            try:
                inds.append(self.vdict[w])
            except:
                pass

        if len(inds) > 0:
            # compute vector representation for question text
            av = np.average(L[:, inds], axis=1).reshape((self.d, 1))
            p = relu(np.dot(W, av) + b)
            p2 = relu(np.dot(W2, p) + b2)
            p3 = relu(np.dot(W3, p2) + b3)

        else:
            p3 = np.zeros((self.d, 1))

        curr_feats = {}
        for dim, val in np.ndenumerate(p3):
            curr_feats[str(dim[0])] = val

        return curr_feats, text

    # return a distribution over answers for the given question
    def compute_probs(self, text):
        W, b, W2, b2, W3, b3, L = self.params

        # generate word vector lookups given normalized input text
        text = normalize(text)
        for ner in self.ners:
            text = text.replace(ner, ner.replace(' ', '_'))

        inds = []
        for w in text.split():
            try:
                inds.append(self.vdict[w])
            except:
                pass

        if len(inds) > 0:
            # compute vector representation for question text
            av = np.average(L[:, inds], axis=1).reshape((self.d, 1))
            p = relu(np.dot(W, av) + b)
            p2 = relu(np.dot(W2, p) + b2)
            p3 = relu(np.dot(W3, p2) + b3)

        else:
            p3 = np.zeros((self.d, 1))

        curr_feats = {}
        for dim, val in np.ndenumerate(p3):
            curr_feats['__' + str(dim)] = val

        preds = self.classifier.prob_classify(curr_feats)
        return preds

    # return top n guesses for a given question
    def text_guess(self, text):
        text = ' '.join(text)
        preds = self.compute_probs(text)
        c = Counter()
        for k, v in preds._prob_dict.items():
            c[k] = v

        res = {}
        for k, v in c.most_common(N_GUESSES):
            try:
                res[self.page_dict[self.vocab[k]]] = v
            except KeyError:
                # Workaround for odd unicode issues (Jordan)
                try:
                    html_parser
                except NameError:
                    import HTMLParser
                    html_parser = HTMLParser.HTMLParser()
                replace_html = html_parser.unescape(
                    self.vocab[k].encode('ascii', 'xmlcharrefreplace'))
                if replace_html not in self.page_dict:
                    print("Missing: %s" % replace_html)
                else:
                    res[self.page_dict[replace_html]] = v

        return res

    def score_one_guess(self, title, text):

        if isinstance(text, list):
            text = ' '.join(text)

        # lowercase and concatenate title words
        title = title.lower().replace(' ', '_')

        preds = self.compute_probs(text)

        # return -1 if classifier doesn't recognize the given guess
        try:
            guess_ind = self.vdict[title]
            val = preds.prob(guess_ind)
        except:
            val = -1

        return {0: val}

    def vw_from_title(self, title, text):
        val = self.score_one_guess(title, text)
        return self.vw_from_score(val)

    def vw_from_score(self, val):
        val = val[list(val.keys())]
        res = "|%s" % self.name
        if val == -1:
            res += " deepfound:0 deepscore:0.0"
        else:
            res += " deepfound:1 deepscore:%f" % val

        return res
