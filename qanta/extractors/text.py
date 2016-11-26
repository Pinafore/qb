from qanta.extractors.abstract import AbstractFeatureExtractor
from qanta.util.constants import ALPHANUMERIC


class TextExtractor(AbstractFeatureExtractor):
    def __init__(self):
        super(TextExtractor, self).__init__()
        self.name = 'text'

    def score_guesses(self, guesses, text):
        line = "|text %s" % ALPHANUMERIC.sub(' ', text.lower())
        for _ in guesses:
            yield line
