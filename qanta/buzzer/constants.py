import os
from collections import namedtuple
from qanta.util import constants as c

N_GUESSES = 50
MIN_ANSWERS = 1
MIN_SCORE = 0
GUESSERS = ['DANGuesser', 'ElasticSearchGuesser']
N_GUESSERS = len(GUESSERS)

BuzzStats = namedtuple('BuzzStats', ['num_total', 'num_hopeful', 'reward',
    'reward_hopeful', 'buzz', 'correct', 'rush', 'late'])

OPTIONS_DIR = 'output/buzzer/options.pkl'
GUESSES_DIR = os.path.join(c.GUESSER_TARGET_PREFIX, 'merged')
