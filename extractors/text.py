# -*- coding: utf-8 -*-

import re
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
        super(TextExtractor, self).__init__()
        self.name = 'text'

    def vw_from_title(self, title, text):
        return "|text %s" % alphanum.sub(' ', unidecode(text.lower()))

