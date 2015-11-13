from __future__ import absolute_import
from unidecode import unidecode
from extractors.abstract import FeatureExtractor
from util.constants import ALPHANUMERIC


class TextExtractor(FeatureExtractor):
    def __init__(self):
        super(TextExtractor, self).__init__()
        self.name = 'text'

    def vw_from_title(self, title, text):
        return "|text %s" % ALPHANUMERIC.sub(' ', unidecode(text.lower()))

    def features(self, question, candidate):
        pass

    def guesses(self, question):
        pass

    def vw_from_score(self, results):
        pass

