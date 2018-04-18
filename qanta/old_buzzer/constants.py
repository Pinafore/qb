import os
from qanta.util import constants as c

MIN_ANSWERS = 1

OPTIONS_DIR = 'output/buzzer/options.pkl'
GUESSES_DIR = os.path.join(c.GUESSER_TARGET_PREFIX, 'merged')
WORDVEC_DIR = 'data/external/GoogleNews-vectors-negative300.bin'
WORDVEC_DIM = 300

BUZZES_DIR = 'output/buzzer/{0}_buzzes_{1}.pkl'
# PROTOBOWL_DIR = 'data/external/qanta-4may17.log.json'
PROTOBOWL_DIR = 'data/external/qanta-buzz.log'

GUESSER_ACC_POS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
GUESSER_ACC = [0.0, 0.4469487280875181, 0.7067352049558455, 0.8356399103730064,
  0.8924476077500989, 0.9260577303281929]
