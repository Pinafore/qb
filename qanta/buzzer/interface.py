import sys
import time
import pickle
import numpy as np
import codecs
import pandas as pd
import itertools
from functools import partial
from multiprocessing import Pool, Manager
from typing import List, Dict, Tuple, Optional

from qanta.util import constants as c
from qanta.util.io import safe_path
from qanta.config import conf
from qanta.datasets.quiz_bowl import QuestionDatabase
from qanta.guesser.abstract import AbstractGuesser
from qanta.buzzer.util import GUESSERS
from qanta.util.multiprocess import _multiprocess
from qanta import logging

N_GUESSERS = len(GUESSERS)
log = logging.get(__name__)


def _buzzer2vwexpo(buzzes: Dict[int, List[List[float]]], 
        inputs: tuple) -> Tuple[list, list, list, list]:
    '''Multiprocessing worker for buzzer2vwexpo
    buzzes: dictionary of qnum -> buzzing position
    inputs: (qnum, question), queue:
        qnum: int, question id
        question: pd.group, the corresponding guesses
        queue: multiprocessing queue for tracking progress
    return:
        buzzf: list of buzz file entries
        predf: list of vw pred file entries
        metaf: list of vw meta file entries
        finalf: list of final file entries
    '''
    (qnum, question), queue = inputs
    qnum = int(qnum)
    buzz = buzzes[qnum]
    buzzf, predf, metaf, finalf = [], [], [], []
    final_guesses = [None for _ in GUESSERS]
    for i, (g_class, g_group) in enumerate(question.groupby('guesser')):
        # FIXME there might be missing guesses so the length might vary
        g_group = g_group.groupby(['sentence', 'token'])

        for pos, (sent_token, p_group) in enumerate(g_group):
            sent, token = sent_token
            p_group = p_group.sort_values('score', ascending=False)
            # normalize scores
            _sum = sum(p_group.score)
            scores = [(r.score / _sum, r.guess) for r in p_group.itertuples()]
            final_guesses[i] = scores[0][1]
            for rank, (score, guess) in enumerate(scores):
                if np.argmax(buzz[pos]) == i and rank == 0:
                    buzzing = 1
                else:
                    buzzing = 0
                if isinstance(score, np.float):
                    score = score.tolist()
                # force negative weight for guesses that are not chosen
                weight = score if buzzing else score - 1
                predf.append([weight, qnum, sent, token])
                metaf.append([qnum, sent, token, guess])
                # manually do what csv.DictWriter does
                guess = guess if ',' not in guess else '"' + guess + '"'
                buzzf.append([qnum, sent, token, guess, g_class, buzzing, score])
    final_guess = final_guesses[np.argmax(buzz[-1][:N_GUESSERS])]
    final_guess = final_guess if ',' not in final_guess else '"' + final_guess + '"'
    finalf.append([qnum, final_guess])
    queue.put(qnum)
    return buzzf, predf, metaf, finalf

def buzzer2vwexpo(guesses_df: pd.DataFrame, 
        buzzes: Dict[int, List[List[float]]], fold: str) -> None:
    '''Given buzzing positions, generate vw_pred, vw_meta, buzz and final files
    guesses_df: pd.DataFrame of guesses
    buzzes: dictionary of qnum -> buzzing position
    fold: string indicating the data fold
    '''
    inputs = guesses_df.groupby('qnum')
    worker = partial(_buzzer2vwexpo, buzzes)
    result = _multiprocess(worker, inputs, info='buzzer2vwexpo')
    buzzf, predf, metaf, finalf = list(map(list, zip(*result)))

    with codecs.open(safe_path(c.PRED_TARGET.format(fold)), 'w', 'utf-8') as pred_file, \
         codecs.open(safe_path(c.META_TARGET.format(fold)), 'w', 'utf-8') as meta_file, \
         codecs.open(safe_path(c.EXPO_BUZZ.format(fold)), 'w', 'utf-8') as buzz_file, \
         codecs.open(safe_path(c.EXPO_FINAL.format(fold)), 'w', 'utf-8') as final_file:

        buzz_file.write('question,sentence,word,page,evidence,final,weight\n')
        final_file.write('question,answer\n')
        
        log.info('\n\n[buzzer2vwexpo] writing to files')

        buzz_out = '\n'.join('{0},{1},{2},{3},{4},{5},{6}'.format(*r) for r in
                itertools.chain(*buzzf))
        buzz_file.write(buzz_out)
        log.info('buzz file written')

        final_out = '\n'.join('{0},{1}'.format(*r) for r in
                itertools.chain(*finalf))
        final_file.write(final_out)
        log.info('final file written')

        pred_out = '\n'.join('{0} {1}_{2}_{3}'.format(*r) for r in
                itertools.chain(*predf))
        pred_file.write(pred_out)
        log.info('vw_pred file written')

        meta_out = '\n'.join('{0} {1} {2} {3}'.format(*r) for r in
                itertools.chain(*metaf))
        meta_file.write(meta_out)
        log.info('vw_meta file written')
