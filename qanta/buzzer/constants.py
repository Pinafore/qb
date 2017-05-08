import os
from collections import namedtuple
from qanta.util import constants as c

NUM_GUESSES = 20
MIN_ANSWERS = 1

BuzzStats = namedtuple('BuzzStats', ['num_total', 'num_hopeful', 'reward',
    'reward_hopeful', 'buzz', 'correct', 'rush', 'late'])

OPTIONS_DIR = 'output/buzzer/options.pkl'
GUESSES_DIR = os.path.join(c.GUESSER_TARGET_PREFIX, 'merged')
