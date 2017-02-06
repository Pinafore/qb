from qanta.extractors.abstract import AbstractFeatureExtractor
from qanta.util.constants import ALPHANUMERIC


class TextExtractor(AbstractFeatureExtractor):
    @property
    def name(self):
        return 'text'

    def score_guesses(self, guesses, text):
        line = "|text %s" % ALPHANUMERIC.sub(' ', text.lower())
        for _ in guesses:
            yield line
