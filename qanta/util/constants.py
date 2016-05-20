from functools import lru_cache
from collections import OrderedDict
import re
import sys
import unicodedata
import string

from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordTokenizer

NEG_INF = float('-inf')
PAREN_EXPRESSION = re.compile(r'\s*\([^)]*\)\s*')
ENGLISH_STOP_WORDS = set(stopwords.words('english'))
QB_STOP_WORDS = {"10", "ten", "points", "tenpoints", "one", "name", ",", ")", "``", "(", '"', ']',
                 '[', ":", "due", "!", "'s", "''", 'ftp'}
STOP_WORDS = ENGLISH_STOP_WORDS | QB_STOP_WORDS
ALPHANUMERIC = re.compile(r'[\W_]+')
PUNCTUATION = string.punctuation
GRANULARITIES = ["sentence"]
FOLDS = ["dev", "devtest", "test"]
FOLDS_NON_NAQT = ["dev", "test"]

LABEL = 'label'
IR = 'ir'
LM = 'lm'
MENTIONS = 'mentions'
DEEP = 'deep'
ANSWER_PRESENT = 'answer_present'
CLASSIFIER = 'classifier'
WIKILINKS = 'wikilinks'
COUNTRY_LIST_PATH = 'data/country_list.txt'
CLM_PATH = 'data/lm.txt'
WHOOSH_WIKI_INDEX_PATH = 'data/whoosh/wiki'

# Do not change order, it matters for writing in correct order
FEATURE_NAMES = [LABEL, LM, MENTIONS, DEEP, ANSWER_PRESENT, CLASSIFIER, WIKILINKS]
NEGATIVE_WEIGHTS = [16]
# NEGATIVE_WEIGHTS = [2, 4, 8, 16, 32, 64]
MIN_APPEARANCES = 5
N_GUESSES = 400
FEATURES = OrderedDict([
    (LM, None),
    (DEEP, None),
    (ANSWER_PRESENT, None),
    (CLASSIFIER, None),
    (WIKILINKS, None),
    (MENTIONS, None)
])

COMPUTE_OPT_FEATURES = [ANSWER_PRESENT, CLASSIFIER, WIKILINKS, LABEL]
DEEP_OPT_FEATURES = [DEEP]
MEMORY_OPT_FEATURES = [LM, MENTIONS]
LM_OPT_FEATURES = [LM]
MENTIONS_OPT_FEATURES = [MENTIONS]


@lru_cache(maxsize=None)
def get_treebank_tokenizer():
    return TreebankWordTokenizer().tokenize


@lru_cache(maxsize=None)
def get_punctuation_table():
    return dict.fromkeys(
        i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))
