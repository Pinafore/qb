import sys
import time
import pickle
import codecs
import pandas as pd
import itertools
from functools import partial
from functional import seq
from multiprocessing import Pool, Manager
from typing import List, Dict, Tuple, Optional

from qanta.util import constants as c
from qanta.config import conf
from qanta.datasets.quiz_bowl import QuestionDatabase
from qanta.guesser.abstract import AbstractGuesser
from qanta import logging

log = logging.get(__name__)

def _buzzer2vwexpo(buzzes: Dict[int, Tuple[int, int]], 
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
    buzz_pos, buzz_guesser = buzzes[qnum]
    buzzf, predf, metaf, finalf = [], [], [], []
    for i, (g, guesser_group) in enumerate(question.groupby('guesser', sort=True)):
        guesser_group = guesser_group.groupby(['sentence', 'token'], sort=True)
        for pos, (sent_token, group) in enumerate(guesser_group):
            sent, token = sent_token
            group = group.sort_values('score', ascending=False)
            for rank, x in enumerate(group.itertuples()):
                final = int((rank == 0) and (pos == buzz_pos) and i == buzz_guesser)
                score = x.score.tolist()
                # force negative weight for guesses that are not chosen
                weight = score if final else score - 1
                predf.append([weight, qnum, sent, token])
                metaf.append([qnum, sent, token, x.guess])
                guess = x.guess if ',' not in x.guess else '"' + x.guess + '"'
                buzzf.append([qnum, sent, token, guess, x.guesser, final, score])
                if final:
                    finalf.append([x.qnum, guess])
    queue.put(qnum)
    return buzzf, predf, metaf, finalf

def buzzer2vwexpo(guesses_df: pd.DataFrame, 
        buzzes: Dict[int, Tuple[int, int]], fold: str) -> None:
    '''Given buzzing positions, generate vw_pred, vw_meta, buzz and final files
    guesses_df: pd.DataFrame of guesses
    buzzes: dictionary of qnum -> buzzing position
    fold: string indicating the data fold
    '''
    pool = Pool(conf['buzzer']['n_cores'])
    manager = Manager()
    queue = manager.Queue()
    inputs = [(question, queue) for question in guesses_df.groupby('qnum')]
    total_size = len(inputs)
    worker = partial(_buzzer2vwexpo, buzzes)
    result = pool.map_async(worker, inputs)

    # monitor loop
    while True:
        if result.ready():
            break
        else:
            size = queue.qsize()
            sys.stderr.write('\rbuzzer2vwexpo done: {0}/{1}'.format(
                size, total_size))
            time.sleep(0.1)
    sys.stderr.write('\n')

    result = result.get()
    buzzf, predf, metaf, finalf = list(map(list, zip(*result)))

    with codecs.open(safe_path(c.PRED_TARGET.format(fold)), 'w', 'utf-8') as pred_file, \
         codecs.open(safe_path(c.META_TARGET.format(fold)), 'w', 'utf-8') as meta_file, \
         codecs.open(safe_path(c.EXPO_BUZZ.format(fold)), 'w', 'utf-8') as buzz_file, \
         codecs.open(safe_path(c.EXPO_FINAL.format(fold)), 'w', 'utf-8') as final_file:

        buzz_file.write('question,sentence,word,page,evidence,final,weight\n')
        final_file.write('question,answer\n')
        
        log.info('\n\n[buzzer2vwexpo] writing to files')

        buzz_out = '\n'.join('{0},{1},{2},{3}{4}{5}{6}'.format(*r) for r in
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
