import re
import string

from nltk.corpus import stopwords

NEG_INF = float('-inf')
PAREN_EXPRESSION = re.compile(r'\s*\([^)]*\)\s*')
ENGLISH_STOP_WORDS = set(stopwords.words('english'))
QB_STOP_WORDS = {"10", "ten", "points", "tenpoints", "one", "name", ",", ")", "``", "(", '"', ']',
                 '[', ":", "due", "!", "'s", "''", 'ftp'}
STOP_WORDS = ENGLISH_STOP_WORDS | QB_STOP_WORDS
ALPHANUMERIC = re.compile(r'[\W_]+')
PUNCTUATION = string.punctuation
GUESSER_REPORT_FOLDS = ['dev', 'test']
BUZZ_FOLDS = ['dev', 'test', 'expo']
ALL_FOLDS = ['train', 'dev', 'test']

STATS = 'stats'
IR = 'ir'
LM = 'lm'
MENTIONS = 'mentions'
DEEP = 'deep'
ANSWER_PRESENT = 'answer_present'
CLASSIFIER = 'classifier'
WIKILINKS = 'wikilinks'
GUESSERS = 'guessers'

FEATURE_NAMES = [STATS, LM, ANSWER_PRESENT, CLASSIFIER, GUESSERS]

FAST_FEATURES = [ANSWER_PRESENT, STATS]
COMPUTE_OPT_FEATURES = [CLASSIFIER]
DEEP_OPT_FEATURES = [DEEP]
LM_OPT_FEATURES = [LM]
MENTIONS_OPT_FEATURES = [MENTIONS]

COUNTRY_LIST_PATH = 'data/internal/country_list.txt'

SENTENCE_STATS = 'output/guesser/sentence_stats.pickle'

CLM_PATH = 'output/language_model'
CLM_TARGET = 'output/language_model.txt'

GLOVE_WE = 'data/external/deep/glove.6B.300d.txt'

GUESSER_TARGET_PREFIX = 'output/guesser'

QB_SOURCE_LOCATION = 'data/internal/source'

WHOOSH_WIKI_INDEX_PATH = 'output/whoosh/wiki/'
CLASSIFIER_TYPES = ['ans_type', 'category', 'gender']
CLASSIFIER_PICKLE_PATH = 'output/classifier/{0}/{0}.pkl'
CLASSIFIER_GUESS_PROPS = 'output/classifier/guess_properties.pkl'
CLASSIFIER_REPORT_PATH = 'output/reporting/classifier_{}.pdf'
COMPARE_GUESSER_REPORT_PATH = 'output/guesser/guesser_comparison_report_{}.pdf'

PRED_TARGET = 'output/predictions/{0}.pred'
META_TARGET = 'output/vw_input/{0}.meta'

EXPO_BUZZ = 'output/expo/{}.buzz'
EXPO_FINAL = 'output/expo/{}.final'
EXPO_QUESTIONS = 'output/expo/{}.questions.csv'

KEN_LM = 'output/kenlm.binary'

VW_INPUT = 'output/vw_input/{0}.vw.txt'
VW_MODEL = 'output/models/model.vw'
VW_PREDICTIONS = 'output/predictions/{0}.pred'
VW_AUDIT = 'output/predictions/{0}.audit'
VW_META = 'output/vw_input/{0}.meta'
VW_AUDIT_REGRESSOR = 'output/reporting/vw_audit_regressor.txt'
VW_AUDIT_REGRESSOR_CSV = 'output/reporting/vw_audit_regressor.csv'
VW_AUDIT_REGRESSOR_REPORT = 'output/reporting/audit_regressor.pdf'
