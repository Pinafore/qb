import string

from nltk.corpus import stopwords

ENGLISH_STOP_WORDS = set(stopwords.words('english'))
QB_STOP_WORDS = set(["10", "ten", "points", "tenpoints", "one", "name",
                     ",", ")", "``", "(", '"', ']',
                     '[', ":", "due", "!", "'s", "''", 'ftp'])
STOP_WORDS = ENGLISH_STOP_WORDS | QB_STOP_WORDS

PUNCTUATION = string.punctuation
GUESSER_REPORT_FOLDS = ['dev', 'test']
BUZZ_FOLDS = ['dev', 'test', 'expo']
ALL_FOLDS = ['train', 'dev', 'test']

COUNTRY_LIST_PATH = 'data/internal/country_list.txt'

SENTENCE_STATS = 'output/guesser/sentence_stats.pickle'

CLM_PATH = 'output/language_model'
CLM_TARGET = 'output/language_model.txt'
CLM_ORDER = 3
CLM_VOCAB = 100000
CLM_COMPARE = 5
CLM_MAX_SPAN = 5
CLM_HASH_NAMES = False
CLM_SLOP = 0
CLM_LOG_LENGTH = True
CLM_GIVE_SCORE = False
CLM_CUTOFF = -2
CLM_START_RANK = 200
CLM_MIN_SPAN = 2
CLM_CENSOR_SLOP = True
# Longest question we can score
CLM_MAX_LENGTH = 5000
CLM_UNK_TOK = "UNK"
CLM_START_TOK = "<S>"
CLM_END_TOK = "</S>"
CLM_SLOP_TOK = "<SLOP>"
CLM_INT_WORDS = False
CLM_USE_C_VERSION = False

GLOVE_WE = 'data/external/deep/glove.6B.300d.txt'

GUESSER_TARGET_PREFIX = 'output/guesser'

QB_SOURCE_LOCATION = 'data/internal/source'

COMPARE_GUESSER_REPORT_PATH = 'output/guesser/guesser_comparison_report_{}.pdf'

PRED_TARGET = 'output/predictions/{0}.pred'
META_TARGET = 'output/vw_input/{0}.meta'

<<<<<<< HEAD
EXPO_QUESTIONS = 'data/internal/expo.csv'
EXPO_BUZZ = 'output/expo/{}.{}.buzz'
EXPO_FINAL = 'output/expo/{}.{}.final'
=======
EXPO_BUZZ = 'output/expo/{}.buzz'
EXPO_FINAL = 'output/expo/{}.final'
EXPO_QUESTIONS = 'output/expo/{}.questions.csv'
>>>>>>> 10fe75cba4427b5457cbdcf1ea34263b1c5d3a16

KEN_LM = 'output/kenlm.binary'

DOMAIN_TARGET_PREFIX = 'output/deep/domain_data'
DOMAIN_MODEL_FORMAT = 'output/deep/domain_clf{}.vw'
DOMAIN_PREDICTIONS_PREFIX = 'output/deep/domain_preds'
DOMAIN_OUTPUT = 'output/deep/filtered_domain_data'
