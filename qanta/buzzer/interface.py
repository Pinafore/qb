import csv
import time
import codecs
import pickle
import pandas as pd
import itertools
from functools import partial
from functional import seq

from qanta.util import constants as c
from qanta.datasets.quiz_bowl import QuestionDatabase
from qanta.guesser.abstract import AbstractGuesser
from qanta import logging

from util import *
from performance import Prediction, Meta
from multiprocessing import Pool

log = logging.get(__name__)

def process_vw_row(line):
    line = line.split('|')
    qid, sent, token = map(int, line[0].split('\'')[1].split('_'))
    features = dict()
    for section in line[1:-1]:
        section = section.strip().split(' ')
        section_name = section[0]
        for feature in section[1:]:
            feature_name, score = feature.split(':')
            feature_name = section_name + '^' + feature_name
            score = float(score)
            features[feature_name] = score
    guesser = line[-1].strip().split(' ')
    guess = guesser[1]
    scores = dict()
    for feature in guesser[2:]:
        feature_name, score = feature.split(':')
        score = float(score)
        scores[feature_name] = score
    features['guess'] = (guess, scores['DAN_score'])
    return qid, sent, token, guess, features, scores['DAN_score']

def process_question(option2id, all_questions, qnum_q):
    qnum, q = qnum_q
    answer = format_guess(all_questions[qnum].page)

    guess_vecs = []
    results = []
    prev_vec = [0 for _ in range(NUM_GUESSES)]
    prev_dict = {}
    for sent_token, group in q.groupby(['sentence', 'token'], sort=True):
        group = group.sort_values('score', ascending=False)[:NUM_GUESSES]

        # check if top guess is correct
        top_guess = group.guess.tolist()[0]
        results.append(int(top_guess == answer))

        # get the current input vector
        curr_vec = group.score.tolist()
        curr_dict = {x.guess: x.score for x in group.itertuples()}
        diff_vec, isnew_vec = [], []
        for i, x in enumerate(group.itertuples()):
            if x.guess not in prev_dict:
                diff_vec.append(x.score)
                isnew_vec.append(1)
            else:
                diff_vec.append(x.score - prev_dict[x.guess])
                isnew_vec.append(0)
        vec = curr_vec + prev_vec + diff_vec + isnew_vec
        assert(len(vec) == 4 * NUM_GUESSES)
        guess_vecs.append(vec)
        prev_vec = curr_vec
        prev_duct = curr_dict

    return QuestionGuesses(qnum, option2id[answer], guess_vecs, results)

def main1():
    cfg = config()
    id2option = pickle.load(open(cfg.options_dir, 'rb'))
    option2id = {o: i for i, o in enumerate(id2option)}
    question_db = QuestionDatabase()
    all_questions = question_db.all_questions()
    vw_input_df = dict()
    guesses = dict()
    for fold in ['dev', 'test']:
        vw_input = open(c.VW_INPUT.format(fold))
        vw_rows = list(map(process_vw_row, vw_input.readlines()))
        vw_rows = pd.DataFrame(vw_rows, columns=['qnum', 'sentence', 'token',
            'guess', 'features', 'score'])
        vw_input_df[fold] = vw_rows
        worker = partial(process_question, option2id, all_questions)
        guesses[fold] = list(map(worker, vw_rows.groupby('qnum')))
    pickle.dump(vw_input_df, open('vw_input_df.pkl', 'wb'))
    pickle.dump(guesses, open('guesses.pkl', 'wb'))
    return vw_input_df, guesses

def _2predmeta(buzzes, inputs):
    (q, question), queue = inputs
    question = question.groupby(['sentence', 'token'], sort=True)
    preds, metas = [], []
    for pos, (sent_token, group) in enumerate(question):
        sent, token = sent_token
        x = group.sort_values('score', ascending=False).iloc[0]
        final = pos == buzzes[x.qnum]
        preds.append(Prediction(x.score, x.qnum, sent, token))
        metas.append(Meta(x.qnum, sent, token, x.guess))
    queue.put(q)
    return preds, metas

def buzzer2predsmetas(vw_input, buzzes):
    pool = Pool(16)
    manager = Manager()
    queue = manager.Queue()
    inputs = [(question, queue) for question in vw_input.groupby('qnum')]
    total_size = len(inputs)
    worker = partial(_2predmeta, buzzes)
    result = pool.map_async(worker, inputs)
    # monitor loop
    while True:
        if result.ready():
            break
        else:
            size = queue.qsize()
            sys.stderr.write('\r[interface] done: {0}/{1}'.format(size, total_size))
            time.sleep(0.1)

    result = result.get()
    preds, metas = list(map(list, zip(*result)))
    preds = list(itertools.chain(*preds))
    metas = list(itertools.chain(*metas))
    return seq(preds), seq(metas)

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
