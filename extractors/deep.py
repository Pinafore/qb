from __future__ import absolute_import
from future.builtins import range, chr, str

import re
from util.imports import pickle
import sys
import unicodedata
import numpy as np
from string import ascii_lowercase, punctuation
from collections import defaultdict, Counter

from unidecode import unidecode

from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.corpus import stopwords

from extractors.abstract import FeatureExtractor
from util.qdb import QuestionDatabase
from util.constants import PAREN_EXPRESSION

QB_STOP = {"10", "ten", "points", "name", ",", ")", "(", '"', ']', '[', ":", "ftp"}

tokenizer = TreebankWordTokenizer().tokenize
stopwords = set(stopwords.words('english')) | QB_STOP
valid_strings = set(ascii_lowercase) | set(str(x) for x in range(10)) | {' '}
punct_tbl = dict.fromkeys(i for i in range(sys.maxunicode)
                          if unicodedata.category(chr(i)).startswith('P'))


def relu(x):
    return x * (x > 0)


def normalize(text):
    text = unidecode(text).lower().translate(str.maketrans(punctuation, ' ' * len(punctuation)))
    text = PAREN_EXPRESSION.sub("", text)
    text = " ".join(x for x in text.split() if x not in stopwords)
    return ''.join(x for x in text if x in valid_strings)


class DeepExtractor(FeatureExtractor):
    def __init__(self, classifier, params, vocab, ners, page_dict, num_results=200):
        super(DeepExtractor, self).__init__()
        self.classifier = pickle.load(open(classifier, 'rb'))
        self.params = pickle.load(open(params, 'rb'))
        self.d = self.params[-1].shape[0]
        self.vocab, self.vdict = pickle.load(open(vocab, 'rb'))
        self.ners = pickle.load(open(ners, 'rb'))
        self.page_dict = page_dict
        self.name = "deep"
        self._limit = num_results

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

        res = defaultdict(dict)
        for k, v in c.most_common(self._limit):
            try:
                res[self.page_dict[self.vocab[k]]][0] = v
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
                    res[self.page_dict[replace_html]][0] = v

        return res

    # return softmax probability of title given text
    ## to-do (mohit): cache probdist for given text, so if we've already computed
    ##                it then we can just look it up later
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
        val = val[val.keys()[0]]
        res = "|%s" % self.name
        if val == -1:
            res += " deepfound:0 deepscore:0.0"
        else:
            res += " deepfound:1 deepscore:%f" % val

        return res


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Demo for deep guesser")
    parser.add_argument("--classifier", default="data/deep/classifier",
                        help="Location of classifier pickle")
    parser.add_argument("--params", default="data/deep/params.pkl",
                        help="Location of parameter pickle")
    parser.add_argument("--vocab", default="data/deep/deep_vocab.pkl",
                        help="Location of vocab pickle")
    parser.add_argument("--ners", default="data/common/ners.pkl",
                        help="Location of NER pickle")
    flags = parser.parse_args()

    import time

    start = time.time()

    questions = questions = QuestionDatabase("data/questions.db")
    page_dict = {}
    for page in questions.get_all_pages():
        page_dict[page.lower().replace(' ', '_')] = page
    ws = DeepExtractor(flags.classifier, flags.params, flags.vocab, flags.ners,
                       page_dict)

    print("Startup: %f sec" % (time.time() - start))

    tests = {}
    tests[u"Tannhäuser (opera)"] = u"""He sought out the pope to
    seek forgiveness of his sins, only to be told that just as the pope's staff
    would never (*) blossom, his sins are never be forgiven. Three days later,
    the pope's staff miraculously bore flowers. For 10 points--identify this
    German folk hero, the subject of an opera by Wagner [VAHG-ner]."""

    guesses = ["Arkansas", "Australia", u"Tannhäuser (opera)", "William Shakespeare"]

    for ii in tests:
        print(ii)

        for gg in guesses:
            start = time.time()
            score = ws.score_one_guess(gg, tests[ii])
            end = time.time()
            print("Score for %s: %f computed in %f seconds" %
                  (gg, score, end-start))
