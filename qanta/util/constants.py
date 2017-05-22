import string

from nltk.corpus import stopwords

ENGLISH_STOP_WORDS = set(stopwords.words('english'))
QB_STOP_WORDS = {"10", "ten", "points", "tenpoints", "one", "name", ",", ")", "``", "(", '"', ']',
                 '[', ":", "due", "!", "'s", "''", 'ftp'}
STOP_WORDS = ENGLISH_STOP_WORDS | QB_STOP_WORDS

PUNCTUATION = string.punctuation
ALL_FOLDS = [
    # Guessers should train and cross validate on these folds
    'guesstrain', 'guessdev',
    # Guessers should produce output for these, only buzzer should train and cross validate on these
    'buzzertrain', 'buzzerdev',
    # Parameter tuning of system should be done on dev. This should be reserved for system-wide parameters and not as
    # a second buzzerdev or guessdev fold test should be reserved for final paper results
    'dev', 'test',
    # Produce output for anything in the expo fold
    'expo'
]

# Training folds
GUESSER_TRAIN_FOLD = 'guesstrain'
BUZZER_TRAIN_FOLD = 'buzzertrain'

# Guesser and buzzers produce reports on these for cross validation
GUESSER_DEV_FOLD = 'guessdev'
BUZZER_DEV_FOLD = 'buzzerdev'

# System-wide cross validation and testing
SYSTEM_DEV_FOLD = 'dev'
SYSTEM_TEST_FOLD = 'test'
EXPO_FOLD = 'expo'

# Guessers should produce test-time guesses on these
GUESSER_GENERATION_FOLDS = [
    GUESSER_DEV_FOLD,
    BUZZER_TRAIN_FOLD, BUZZER_DEV_FOLD,
    SYSTEM_DEV_FOLD, SYSTEM_TEST_FOLD,
    EXPO_FOLD
]

BUZZER_INPUT_FOLDS = [
    BUZZER_TRAIN_FOLD, BUZZER_DEV_FOLD,
    SYSTEM_DEV_FOLD, SYSTEM_TEST_FOLD,
    EXPO_FOLD
]

# Buzzers should produce test-time guesses on these
BUZZER_GENERATION_FOLDS = [
    BUZZER_DEV_FOLD,
    SYSTEM_DEV_FOLD, SYSTEM_TEST_FOLD,
    EXPO_FOLD
]


BUZZ_FOLDS = ['dev', 'test', 'expo']

WIKI_LOCATION = 'data/external/wikipedia'

ALL_WIKI_REDIRECTS = 'data/external/wikipedia/all_wiki_redirects.csv'
WIKI_DUMP_REDIRECT_PICKLE = 'data/external/wikipedia/dump_redirects.pickle'

COUNTRY_LIST_PATH = 'data/internal/country_list.txt'

SENTENCE_STATS = 'output/guesser/sentence_stats.pickle'

CLM_PATH = 'output/language_model'
CLM_TARGET = 'output/language_model.txt'

GLOVE_WE = 'data/external/deep/glove.6B.300d.txt'

GUESSER_TARGET_PREFIX = 'output/guesser'

QB_SOURCE_LOCATION = 'data/internal/source'

COMPARE_GUESSER_REPORT_PATH = 'output/guesser/guesser_comparison_report_{}.pdf'

PRED_TARGET = 'output/predictions/{0}.pred'
META_TARGET = 'output/vw_input/{0}.meta'

EXPO_BUZZ = 'output/expo/{}.buzz'
EXPO_FINAL = 'output/expo/{}.final'
EXPO_QUESTIONS = 'output/expo/{}.questions.csv'

KEN_LM = 'output/kenlm.binary'

DOMAIN_TARGET_PREFIX = 'output/deep/domain_data'
DOMAIN_MODEL_FORMAT = 'output/deep/domain_clf{}.vw'
DOMAIN_PREDICTIONS_PREFIX = 'output/deep/domain_preds'
DOMAIN_OUTPUT = 'output/deep/filtered_domain_data'
