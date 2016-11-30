from functools import lru_cache
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
VW_FOLDS = ['dev', 'test']
ALL_FOLDS = ['train', 'dev', 'test']

STATS = 'stats'
IR = 'ir'
LM = 'lm'
MENTIONS = 'mentions'
DEEP = 'deep'
ANSWER_PRESENT = 'answer_present'
CLASSIFIER = 'classifier'
WIKILINKS = 'wikilinks'
TEXT = 'text'
GUESSERS = 'guessers'

FEATURE_NAMES = [STATS, LM, ANSWER_PRESENT, CLASSIFIER, TEXT, GUESSERS]

FAST_FEATURES = [ANSWER_PRESENT, STATS, TEXT]
COMPUTE_OPT_FEATURES = [CLASSIFIER]
DEEP_OPT_FEATURES = [DEEP]
LM_OPT_FEATURES = [LM]
MENTIONS_OPT_FEATURES = [MENTIONS]

GUESSER_LIST = [
    ('qanta.guesser.random.RandomGuesser', 'qanta.pipeline.guesser.random.EmptyTask')
]

COUNTRY_LIST_PATH = 'data/internal/country_list.txt'

SENTENCE_STATS = 'output/guesser/sentence_stats.pickle'

NERS_PATH = 'data/internal/common/ners'

CLM_PATH = 'output/language_model'
CLM_TARGET = 'output/language_model.txt'

DEEP_WE_TARGET = 'output/deep/We'
DEEP_DAN_PARAMS_TARGET = 'output/deep/params'
DEEP_DAN_CLASSIFIER_TARGET = 'output/deep/classifier'
DEEP_TRAIN_TARGET = 'output/deep/train'
DEEP_DEV_TARGET = 'output/deep/dev'
DEEP_DAN_TRAIN_OUTPUT = 'output/deep/train_dan'
DEEP_DAN_DEV_OUTPUT = 'output/deep/dev_dan'
DEEP_DEVTEST_TARGET = 'output/deep/devtest'
DEEP_TEST_TARGET = 'output/deep/test'
DEEP_VOCAB_TARGET = 'output/deep/vocab'
EVAL_RES_TARGET = 'output/deep/eval_res'

GUESSER_TARGET_PREFIX = 'output/guesser'

WIKIFIER_INPUT_TARGET = 'output/wikifier/input'
WIKIFIER_OUTPUT_TARGET = 'output/wikifier/output'

QB_SOURCE_LOCATION = 'data/internal/source'

WHOOSH_WIKI_INDEX_PATH = 'output/whoosh/wiki'
CLASSIFIER_TYPES = ['ans_type', 'category', 'gender']
CLASSIFIER_PICKLE_PATH = 'output/classifier/{0}/{0}.pkl'
CLASSIFIER_REPORT_PATH = 'output/reporting/classifier_{}.pdf'

PRED_TARGET = 'output/predictions/{0}.pred'
META_TARGET = 'output/vw_input/{0}.meta'

EXPO_BUZZ = 'output/expo/{}.{}.buzz'
EXPO_FINAL = 'output/expo/{}.{}.final'
EXPO_QUESTIONS = 'output/expo/test.questions.csv'

KEN_LM = 'output/kenlm.binary'

VW_INPUT = 'output/vw_input/{0}.vw.gz'
VW_MODEL = 'output/models/model.vw'
VW_PREDICTIONS = 'output/predictions/{0}.pred'
VW_AUDIT = 'output/predictions/{0}.audit'
VW_META = 'output/vw_input/{0}.meta'
VW_AUDIT_REGRESSOR = 'output/reporting/vw_audit_regressor.txt'
VW_AUDIT_REGRESSOR_CSV = 'output/reporting/vw_audit_regressor.csv'
VW_AUDIT_REGRESSOR_REPORT = 'output/reporting/audit_regressor.pdf'

NEGATIVE_WEIGHTS = [16]
MIN_APPEARANCES = 5
MAX_APPEARANCES = 5
N_GUESSES = 200


@lru_cache(maxsize=None)
def get_treebank_tokenizer():
    return TreebankWordTokenizer().tokenize


@lru_cache(maxsize=None)
def get_punctuation_table():
    return dict.fromkeys(
        i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))
