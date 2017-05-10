import os
from collections import namedtuple
from qanta.util import constants as c

MIN_ANSWERS = 1
NEG_WEIGHT = 0.2 # weight of negative (not buzzing) class

BuzzStats = namedtuple('BuzzStats', ['num_total', 'num_hopeful', 'reward',
    'reward_hopeful', 'buzz', 'correct', 'rush', 'late'])

OPTIONS_DIR = 'output/buzzer/options.pkl'
GUESSES_DIR = os.path.join(c.GUESSER_TARGET_PREFIX, 'merged')

BUZZER_MODEL = 'output/buzzer/mlp_buzzer.npz'
