import os
from qanta.util import constants as c

MIN_ANSWERS = 1
NEG_WEIGHT = 0.2 # weight of negative (not buzzing) class

OPTIONS_DIR = 'output/buzzer/options.pkl'
GUESSES_DIR = os.path.join(c.GUESSER_TARGET_PREFIX, 'merged')
