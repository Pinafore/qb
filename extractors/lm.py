from extractors.abstract import FeatureExtractor
from clm.lm_wrapper import LanguageModelReader
import requests

kINTERP_CONSTANTS = 0.9


class LanguageModel(FeatureExtractor):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        print("Starting to read the LM from %s" % self.filename)
        self._lm = LanguageModelReader(self.filename)
        self._lm.init()
        self._name = "lm"
        self._corpora = set()

    def add_corpus(self, corpus_name):
        self._corpora.add(corpus_name)

    def vw_from_title(self, title, text):
        return "|%s %s" % (self._name,
                           " ".join(self._lm.feature_line(x, title, text) for
                                    x in self._corpora))

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
        return requests.post('http://localhost/', data={title: title, text: text})
