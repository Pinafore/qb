from feature_extractor import FeatureExtractor
from clm.lm_wrapper import LanguageModelReader

kINTERP_CONSTANTS = 0.9


class LanguageModel(FeatureExtractor):
    def __init__(self, filename):
        self._lm = LanguageModelReader(filename)
        print("Starting to read the LM from %s" % filename)
        self._lm.init()
        self._name = "lm"
        self._corpora = set()

    def add_corpus(self, corpus_name):
        self._corpora.add(corpus_name)

    def vw_from_title(self, title, text):
        return " ".join(self._lm.feature_line(x, title, text) for
                        x in self._corpora)
