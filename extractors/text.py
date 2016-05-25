# -*- coding: utf-8 -*-

import re
import sys

from unidecode import unidecode

from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.corpus import stopwords

from feature_extractor import FeatureExtractor

kNEG_INF = float("-inf")
alphanum = pattern = re.compile('[\W_]+')
tokenizer = TreebankWordTokenizer().tokenize
stopwords = set(stopwords.words('english'))


class TextExtractor(FeatureExtractor):
    def __init__(self):
        self._cache = 0
        self._line = ""

    def vw_from_title(self, title, text):
        if self._cache != hash(text):
            self._cache = hash(text)
            self._line = "|text %s" % alphanum.sub(' ', unidecode(text.lower()))
        return self._line

    def name(self):
        return "text"
