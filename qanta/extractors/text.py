from unidecode import unidecode
from qanta.extractors.abstract import FeatureExtractor
from qanta.util.constants import ALPHANUMERIC


class TextExtractor(FeatureExtractor):
    def __init__(self):
        super(TextExtractor, self).__init__()
        self.name = 'text'

    def score_guesses(self, guesses, text):
        line = "|text %s" % ALPHANUMERIC.sub(' ', unidecode(text.lower()))
        for g in guesses:
            yield line, g