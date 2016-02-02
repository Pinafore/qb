from __future__ import absolute_import
from future.builtins import chr, range
from functools import lru_cache
import re
import sys
import unicodedata
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordTokenizer

NEG_INF = float('-inf')
PAREN_EXPRESSION = re.compile('\s*\([^)]*\)\s*')
ENGLISH_STOP_WORDS = set(stopwords.words('english'))
QB_STOP_WORDS = {"10", "ten", "points", "tenpoints", "one", "name", ",", ")", "``", "(", '"', ']', '[',
            ":", "due", "!", "'s", "''", 'ftp'}
STOP_WORDS = ENGLISH_STOP_WORDS | QB_STOP_WORDS
ALPHANUMERIC = re.compile('[\W_]+')
GRANULARITIES = ["sentence"]
FOLDS = ["dev", "devtest", "test"]


@lru_cache(maxsize=None)
def get_treebank_tokenizer():
    return TreebankWordTokenizer().tokenize


@lru_cache(maxsize=None)
def get_punctuation_table():
    return dict.fromkeys(
        i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))
