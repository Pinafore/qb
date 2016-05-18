# -*- coding: utf-8 -*-
import six
from string import ascii_lowercase, ascii_uppercase, digits
from collections import defaultdict

from numpy import isnan

import whoosh
from whoosh import index
from whoosh import qparser
from whoosh import scoring
from whoosh.collectors import TimeLimitCollector, TimeLimit

from unidecode import unidecode

from qanta.extractors.abstract import FeatureExtractor
from qanta.util.constants import (
    NEG_INF, STOP_WORDS, PAREN_EXPRESSION, get_treebank_tokenizer, get_punctuation_table)
from qanta.util.environment import data_path

QUERY_CHARS = set(ascii_lowercase + ascii_uppercase + digits)

tokenizer = get_treebank_tokenizer()
valid_strings = set(ascii_lowercase) | set(str(x) for x in range(10)) | set(' ')


class IrExtractor(FeatureExtractor):
    def __init__(self):
        super(IrExtractor, self).__init__()
        self.name = "ir"

    def set_metadata(self, answer, category, qnum, sent, token, guesses, fold):
        pass

    def score_one_guess(self, title, text):
        val = {}
        for ii in self._index:
            val[ii] = self._index[ii].score_one_guess(title, text)
        return val

    def vw_from_title(self, title, text):
        pass

    def vw_from_score(self, results):
        pass

    def text_guess(self, text):
        pass
