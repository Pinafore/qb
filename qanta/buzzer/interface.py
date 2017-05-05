import csv
import time
import codecs
import pickle
import pandas as pd
import itertools
from functools import partial
from functional import seq
from multiprocessing import Pool

from qanta.util import constants as c
from qanta.datasets.quiz_bowl import QuestionDatabase
from qanta.guesser.abstract import AbstractGuesser
from qanta import logging

log = logging.get(__name__)

def _buzzer2vwexpo(buzzes, inputs):
    (q, question), queue = inputs
    question = question.groupby(['sentence', 'token'], sort=True)
    q = int(q)
    buzzf, predf, metaf, finalf = [], [], [], []
    for pos, (sent_token, group) in enumerate(question):
        sent, token = sent_token
        group = group.sort_values('score', ascending=False)[:NUM_GUESSES]
        for rank, x in enumerate(group.itertuples()):
            final = int((rank == 0) and (pos == buzzes[x.qnum]))
            score = x.score.tolist()
            weight = score if final else score - 1
            predf.append([weight, q, sent, token])
            metaf.append([q, sent, token, x.guess])
            buzzf.append([q, sent, token, x.guess, '', final, score])
            if final:
                finalf.append([x.qnum, x.guess])
    queue.put(q)
    return buzzf, predf, metaf, finalf


def buzzer2vwexpo(vw_input, buzzes, fold):
    pool = Pool(16)
    manager = Manager()
    queue = manager.Queue()
    inputs = [(question, queue) for question in vw_input.groupby('qnum')]
    total_size = len(inputs)
    worker = partial(_buzzer2vwexpo, buzzes)
    result = pool.map_async(worker, inputs)

    # monitor loop
    while True:
        if result.ready():
            break
        else:
            size = queue.qsize()
            sys.stderr.write('\rbuzzer2vwexpo done: {0}/{1}'.format(size, total_size))
            time.sleep(0.1)

    result = result.get()
    buzzf, predf, metaf, finalf = list(map(list, zip(*result)))

    with open(c.PRED_TARGET.format(fold), 'w') as pred_file, \
         open(c.META_TARGET.format(fold), 'w') as meta_file, \
         open(c.EXPO_BUZZ.format(fold), 'w') as buzz_file, \
         open(c.EXPO_FINAL.format(fold), 'w') as final_file:

        buzz_writer = csv.writer(buzz_file, delimiter=',')
        final_writer = csv.writer(final_file, delimiter=',')
        final_file.write('question,answer\n')
        
        log.info('\n\n[buzzer2vwexpo] writing to files')

        for row in itertools.chain(*buzzf):
            buzz_writer.writerow(row)
        log.info('buzz file written')
        for row in itertools.chain(*finalf):
            final_writer.writerow(row)
        log.info('final file written')
        for row in itertools.chain(*predf):
            pred_file.write('{0} {1}_{2}_{3}\n'.format(*row))
        log.info('vw_pred file written')
        for row in itertools.chain(*metaf):
            meta_file.write('{0} {1} {2} {3}\n'.format(*row))
        log.info('vw_meta file written')
