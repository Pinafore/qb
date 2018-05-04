from os import path

ALL_FOLDS = [
    # Guessers should train and cross validate on these folds
    'guesstrain', 'guessdev',
    # Guessers should produce output for these, only buzzer should train and cross validate on these
    'buzztrain', 'buzzdev',
    # Parameter tuning of system should be done on dev. This should be reserved for system-wide parameters and not as
    # a second buzzerdev or guessdev fold test should be reserved for final paper results
    'dev', 'test',
    # Produce output for anything in the expo fold
    'expo'
]

# Training folds
GUESSER_TRAIN_FOLD = 'guesstrain'
BUZZER_TRAIN_FOLD = 'buzztrain'
TRAIN_FOLDS = {GUESSER_TRAIN_FOLD, BUZZER_TRAIN_FOLD}

# Guesser and buzzers produce reports on these for cross validation
GUESSER_DEV_FOLD = 'guessdev'
BUZZER_DEV_FOLD = 'buzzdev'
SYSTEM_DEV_FOLD = 'dev'
DEV_FOLDS = {GUESSER_DEV_FOLD, BUZZER_DEV_FOLD, SYSTEM_DEV_FOLD}

# System-wide cross validation and testing
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
WIKI_TITLES_PICKLE = 'data/external/wikipedia/wikipedia-titles.pickle'
WIKI_LOOKUP_PATH = 'data/external/wikipedia/wiki_lookup.json'
WIKI_INSTANCE_OF_PICKLE = 'data/external/wikidata_instance-of.pickle'

COUNTRY_LIST_PATH = 'data/internal/country_list.txt'

SENTENCE_STATS = 'output/guesser/sentence_stats.pickle'

GLOVE_WE = 'data/external/deep/glove.6B.300d.txt'

GUESSER_TARGET_PREFIX = 'output/guesser'
GUESSER_REPORTING_PREFIX = 'output/reporting/guesser'

QB_SOURCE_LOCATION = 'data/internal/source'

PRED_TARGET = 'output/predictions/{0}.pred'
META_TARGET = 'output/vw_input/{0}.meta'

EXPO_BUZZ = 'output/expo/{}.buzz'
EXPO_FINAL = 'output/expo/{}.final'
EXPO_QUESTIONS = 'output/expo/{}.questions.csv'

DS_VERSION = '2018.04.18'
DATASET_PREFIX = 'data/external/datasets'
QANTA_MAP_REPORT_PATH = 'data/external/answer_mapping/match_report.json'
QANTA_MAPPED_DATASET_PATH = path.join(DATASET_PREFIX, f'qanta.mapped.{DS_VERSION}.json')
QANTA_SQL_DATASET_PATH = path.join(DATASET_PREFIX, f'qanta.{DS_VERSION}.sqlite3')
QANTA_TRAIN_DATASET_PATH = path.join(DATASET_PREFIX, f'qanta.train.{DS_VERSION}.json')
QANTA_DEV_DATASET_PATH = path.join(DATASET_PREFIX, f'qanta.dev.{DS_VERSION}.json')
QANTA_TEST_DATASET_PATH = path.join(DATASET_PREFIX, f'qanta.test.{DS_VERSION}.json')

QANTA_TORCH_TRAIN = f'qanta.torchtext.train.{DS_VERSION}.json'
QANTA_TORCH_TRAIN_LOCAL_PATH = path.join(DATASET_PREFIX, QANTA_TORCH_TRAIN)
QANTA_TORCH_VAL = f'qanta.torchtext.val.{DS_VERSION}.json'
QANTA_TORCH_VAL_LOCAL_PATH = path.join(DATASET_PREFIX, QANTA_TORCH_VAL)
QANTA_TORCH_DEV = f'qanta.torchtext.dev.{DS_VERSION}.json'
QANTA_TORCH_DEV_LOCAL_PATH = path.join(DATASET_PREFIX, QANTA_TORCH_DEV)
