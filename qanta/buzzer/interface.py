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


def buzzer2expo(vw_input, buzzes):
    '''
    vw_input: dataframe
    buzzes: qid -> buzzing position, -1 if never buzz
    '''
    buzzfile = codecs.open(c.EXPO_BUZZ.format('test'), 'w', 'utf-8')
    buzzfile.write('question,sentence,word,page,evidence,final,weight\n')
    buzzwriter = csv.writer(buzzfile, delimiter=',')

    finalfile = codecs.open(c.EXPO_FINAL.format('test'), 'w', 'utf-8')
    finalfile.write('question,answer\n')
    finalwriter = csv.writer(finalfile, delimiter=',')

    for q, question in vw_input.groupby('qnum'):
        question = question.groupby(['sentence', 'token'], sort=True)
        for pos, (sent_token, group) in enumerate(question):
            group = group.sort_values('score', ascending=False)[:NUM_GUESSES]
            for rank, x in enumerate(group.itertuples()):
                # buzz for the top guess only
                final = int((rank == 0) and (pos == buzzes[x.qnum]))
                buzzwriter.writerow([x.qnum, x.sentence, x.token, x.guess, '',
                    final, x.score])
                # for feature_name, score in x.features.items():
                #     if feature_name != 'guess':
                #         evidence += '{}:{} '.format(feature_name, score)
                # evidence = evidence[:-1]
                if final:
                    finalwriter.writerow([x.qnum, x.guess])


def buzzer2vwout(vw_input, buzzes, fold):
    '''
    vw_input: dataframe
    buzzes: qid -> buzzing position, -1 if never buzz
    '''

    predfile = codecs.open(c.PRED_TARGET.format(fold), 'w', 'utf-8')
    metafile = codecs.open(c.META_TARGET.format(fold), 'w', 'utf-8')

    for q, question in vw_input.groupby('qnum'):
        question = question.groupby(['sentence', 'token'], sort=True)
        q = int(q)
        for pos, (sent_token, group) in enumerate(question):
            sent, token = sent_token
            group = group.sort_values('score', ascending=False)[:NUM_GUESSES]
            for rank, x in enumerate(group.itertuples()):
                final = int((rank == 0) and (pos == buzzes[x.qnum]))
                # negative if not buzzing, but preserve the ranking
                weight = x.score if final else x.score - 1
                predfile.write('{0} {1}_{2}_{3}\n'.format(weight, x.qnum,
                    x.sentence, x.token))
                metafile.write('{0} {1} {2} {3}\n'.format(x.qnum, x.sentence,
                    x.token, x.guess))


def _2vwexpo(buzzes, inputs):
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
    worker = partial(_2vwexpo, buzzes)
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

    # vw_output
    predfile = codecs.open(c.PRED_TARGET.format(fold), 'w', 'utf-8')
    metafile = codecs.open(c.META_TARGET.format(fold), 'w', 'utf-8')

    # expo buzz file
    buzzfile = codecs.open(c.EXPO_BUZZ.format('test'), 'w', 'utf-8')
    buzzfile.write('question,sentence,word,page,evidence,final,weight\n')
    buzzwriter = csv.writer(buzzfile, delimiter=',')

    # expo final file
    finalfile = codecs.open(c.EXPO_FINAL.format('test'), 'w', 'utf-8')
    finalfile.write('question,answer\n')
    finalwriter = csv.writer(finalfile, delimiter=',')
    
    log.info('\n\n[buzzer2vwexpo] writing to files')

    for row in itertools.chain(*buzzf):
        buzzwriter.writerow(row)
    log.info('buzz file written')
    for row in itertools.chain(*finalf):
        finalwriter.writerow(row)
    log.info('final file written')
    for row in itertools.chain(*predf):
        predfile.write('{0} {1}_{2}_{3}\n'.format(*row))
    log.info('vw_pred file written')
    for row in itertools.chain(*metaf):
        metafile.write('{0} {1} {2} {3}\n'.format(*row))
    log.info('vw_meta file written')
