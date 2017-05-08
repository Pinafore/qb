import os
import pandas as pd

from qanta.util import constants as c
from qanta.guesser.abstract import AbstractGuesser
from qanta import logging

log = logging.get(__name__)

guessers = ['qanta.guesser.dan.DANGuesser', 
            'qanta.guesser.elasticsearch.ElasticSearchGuesser']

log.info("Merging guesser DataFrames.")
for fold in ['dev', 'test']:
    new_guesses = pd.DataFrame(columns=['fold', 'guess', 'guesser', 'qnum',
        'score', 'sentence', 'token'], dtype='object')
    for guesser in guessers:
        guesser_dir = os.path.join(c.GUESSER_TARGET_PREFIX, guesser)
        guesses = AbstractGuesser.load_guesses(guesser_dir, folds=[fold])
        new_guesses = new_guesses.append(guesses)
    merged_dir = os.path.join(c.GUESSER_TARGET_PREFIX, 'merged')
    if not os.path.exists(merged_dir):
        os.makedirs(merged_dir)
    AbstractGuesser.save_guesses(new_guesses, merged_dir, folds=[fold])
    log.info("Merging: {0} finished.".format(fold))
