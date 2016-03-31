from extractors.abstract import FeatureExtractor
from clm.lm_wrapper import LanguageModelReader
import requests

kINTERP_CONSTANTS = 0.9


class LanguageModel(FeatureExtractor):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self.initialized = False
        self._lm = None
        self._name = "lm"
        self.name = 'lm'
        self._corpora = set()

    def set_metadata(self, answer, category, qnum, sent, token, guesses, fold):
        super(LanguageModel, self).set_metadata(answer, category, qnum, sent, token, guesses, fold)
        if not self.initialized:
            print("Starting to read the LM from %s" % self.filename)
            self._lm = LanguageModelReader(self.filename)
            self._lm.init()
            self.initialized = True
            self.add_corpus("qb")
            self.add_corpus("wiki")
            self.add_corpus("source")

    def add_corpus(self, corpus_name):
        self._corpora.add(corpus_name)

    def vw_from_title(self, title, text):
        return "|%s %s" % (self._name,
                           " ".join(self._lm.feature_line(x, title, text) for x in self._corpora))

    def features(self, question, candidate):
        pass

    def vw_from_score(self, results):
        pass

    def guesses(self, question):
        pass


class HTTPLanguageModel(FeatureExtractor):
    def guesses(self, question):
        pass

    def features(self, question, candidate):
        pass

    def vw_from_score(self, results):
        pass

    def vw_from_title(self, title, text):
        return requests.post('http://localhost:5000/', data={'title': title, 'text': text}).text
