# -*- coding: utf-8 -*-

import re
from unidecode import unidecode
from extractors.abstract import FeatureExtractor

alphanum = pattern = re.compile('[\W_]+')


class TextExtractor(FeatureExtractor):
    def __init__(self):
        super(TextExtractor, self).__init__()
        self.name = 'text'

    def vw_from_title(self, title, text):
        return "|text %s" % alphanum.sub(' ', unidecode(text.lower()))

    def features(self, question, candidate):
        pass

    def guesses(self, question):
        pass

    def vw_from_score(self, results):
        pass

