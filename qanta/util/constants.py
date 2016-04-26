from functools import lru_cache
from collections import OrderedDict
import re
import sys
import unicodedata
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordTokenizer

NEG_INF = float('-inf')
PAREN_EXPRESSION = re.compile(r'\s*\([^)]*\)\s*')
ENGLISH_STOP_WORDS = set(stopwords.words('english'))
QB_STOP_WORDS = {"10", "ten", "points", "tenpoints", "one", "name", ",", ")", "``", "(", '"', ']',
                 '[', ":", "due", "!", "'s", "''", 'ftp'}
STOP_WORDS = ENGLISH_STOP_WORDS | QB_STOP_WORDS
ALPHANUMERIC = re.compile(r'[\W_]+')
GRANULARITIES = ["sentence"]
FOLDS = ["dev", "devtest", "test"]
FOLDS_NON_NAQT = ["dev", "test"]

LABEL = 'label'
IR = 'ir'
LM = 'lm'
MENTIONS = 'mentions'
DEEP = 'deep'
ANSWER_PRESENT = 'answer_present'
TEXT = 'text'
CLASSIFIER = 'classifier'
WIKILINKS = 'wikilinks'

# Do not change order, it matters for writing in correct order
FEATURE_NAMES = [LABEL, IR, LM, MENTIONS, DEEP, ANSWER_PRESENT, TEXT, CLASSIFIER, WIKILINKS]
NEGATIVE_WEIGHTS = [2., 4., 8., 16., 32., 64.]
MIN_APPEARANCES = 5
FEATURES = OrderedDict([
    (IR, None),
    (LM, None),
    (DEEP, None),
    (ANSWER_PRESENT, None),
    (TEXT, None),
    (CLASSIFIER, None),
    (WIKILINKS, None),
    (MENTIONS, None)
])

COMPUTE_OPT_FEATURES = [IR, DEEP, ANSWER_PRESENT, TEXT, CLASSIFIER, WIKILINKS, LABEL]
MEMORY_OPT_FEATURES = [LM, MENTIONS]

assert sorted(COMPUTE_OPT_FEATURES + MEMORY_OPT_FEATURES) == sorted(FEATURE_NAMES)
assert len(COMPUTE_OPT_FEATURES) + len(MEMORY_OPT_FEATURES) == len(FEATURE_NAMES)


@lru_cache(maxsize=None)
def get_treebank_tokenizer():
    return TreebankWordTokenizer().tokenize


@lru_cache(maxsize=None)
def get_punctuation_table():
    return dict.fromkeys(
        i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))
