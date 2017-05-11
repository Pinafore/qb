import os
from qanta.util import constants as c

MIN_ANSWERS = 1
NEG_WEIGHT = 1 # weight of negative (not buzzing) class

OPTIONS_DIR = 'output/buzzer/options.pkl'
GUESSES_DIR = os.path.join(c.GUESSER_TARGET_PREFIX, 'merged')
WORDVEC_DIR = 'data/external/GoogleNews-vectors-negative300.bin'
WORDVEC_DIM = 300
