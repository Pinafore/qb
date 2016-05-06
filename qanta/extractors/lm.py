from qanta.extractors.abstract import FeatureExtractor
from clm.lm_wrapper import LanguageModelReader


class LanguageModel(FeatureExtractor):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self.initialized = False
        self.lm = None
        self.name = 'lm'
        self.corpora = set()

    def set_metadata(self, answer, category, qnum, sent, token, guesses, fold):
        if not self.initialized:
            print("Starting to read the LM from %s" % self.filename)
            self.lm = LanguageModelReader(self.filename)
            self.lm.init()
            self.initialized = True
            self.add_corpus("qb")
            self.add_corpus("wiki")
            self.add_corpus("source")

    def add_corpus(self, corpus_name):
        self.corpora.add(corpus_name)

    def vw_from_title(self, title, text):
        return "|%s %s" % (self.name,
                           " ".join(self.lm.feature_line(x, title, text) for x in self.corpora))

    def vw_from_score(self, results):
        pass
