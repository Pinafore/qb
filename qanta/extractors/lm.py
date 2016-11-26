from qanta.extractors.abstract import AbstractFeatureExtractor
from qanta.util.environment import data_path
from qanta.util.constants import CLM_PATH
from clm.lm_wrapper import LanguageModelReader


class LanguageModel(AbstractFeatureExtractor):
    def __init__(self):
        super().__init__()
        self.filename = data_path(CLM_PATH)
        self.initialized = False
        self._lm = None
        self._corpora = set()

    def _init_lm(self):
        if not self.initialized:
            print("Starting to read the LM from %s" % self.filename)
            self._lm = LanguageModelReader(self.filename)
            self._lm.init()
            self.initialized = True
            self._add_corpus("qb")
            self._add_corpus("wiki")
            self._add_corpus("source")

    @property
    def name(self):
        return 'lm'

    @property
    def corpora(self):
        self._init_lm()
        return self._corpora

    @property
    def lm(self):
        self._init_lm()
        return self._lm

    def _add_corpus(self, corpus_name):
        self._corpora.add(corpus_name)

    def score_guesses(self, guesses, text):
        for guess in guesses:
            feature = "|%s %s" % (
                self.name,
                " ".join(self.lm.feature_line(x, guess, text) for x in self.corpora))
            yield feature
