import os
import sys
import time
import pickle
import numpy as np
import argparse
from collections import namedtuple
from multiprocessing import Pool, Manager
from functools import partial

from qanta.datasets.quiz_bowl import QuestionDatabase
from qanta.datasets.quiz_bowl import QuizBowlDataset
from qanta.preprocess import format_guess
from qanta.guesser.abstract import AbstractGuesser
from qanta.util import constants as c
from qanta import logging

NUM_GUESSES = 20
MIN_ANSWERS = 1
Batch = namedtuple('Batch', ['qids', 'answers', 'mask', 'vecs', 'results'])
BuzzStats = namedtuple('BuzzStats', ['num_total', 'num_hopeful', 'reward', 'reward_hopeful', 
                                     'buzz', 'correct', 'rush', 'late'])
OPTIONS_DIR = 'output/buzzer/options.pkl'
GUESSES_DIR = 'data/guesses/'

log = logging.get(__name__)

def stupid_buzzer(iterator):

    def _do_one(vecs_results_masks):
        vecs, results, masks = vecs_results_masks
        hopeful = any(results == 1)
        prev_top_score = vecs[0][0]
        prev_variance = np.var(vecs[0])
        for i in range(len(masks)):
            if masks[i] == 0:
                return hopeful, results[i - 1], - 1
            if vecs[i][0] - prev_top_score > 0.3:
                if np.var(vecs[i]) - prev_variance > 0.005:
                    return hopeful, results[i], i
        return hopeful, results[-1], - 1

    num_hopeful = 0
    num_results = 0
    tot_reward = 0
    buzz_dict = dict()
    count = 0
    for i in range(iterator.size):
        batch = iterator.next_batch(np)
        count += len(batch.qids)
        vecs = np.swapaxes(batch.vecs, 0, 1)
        results = batch.results.T
        masks = batch.mask.T
        returns = map(_do_one, zip(vecs, results, masks))
        returns = list(map(list, zip(*returns)))
        hopeful = sum(returns[0])
        results = sum(returns[1])
        buzzes = list(returns[2])
        reward = ((results * 10) - (hopeful - results) * 5) / hopeful if hopeful > 0 else 0
        num_hopeful += hopeful
        num_results += results
        tot_reward += reward
        for qid, buzz in zip(batch.qids, buzzes):
            qid = qid.tolist()
            assert isinstance(qid, int)
            buzz_dict[qid] = buzz
    tot_reward /= iterator.size
    log.info('[stupid]', num_hopeful, num_results, tot_reward)
    return buzz_dict

def _process_question_df(option2id, all_questions, qnum_q_queue):
    (qnum, q), queue = qnum_q_queue

    answer = format_guess(all_questions[qnum].page)
    if answer in option2id:
        answer_id = option2id[answer]
    else:
        answer_id = len(option2id)

    guess_vecs = []
    results = []
    prev_vec = [0 for _ in range(NUM_GUESSES)]
    prev_dict = {}
    for sent_token, group in q.groupby(['sentence', 'token'], sort=True):
        group = group.sort_values('score', ascending=False)[:NUM_GUESSES]

        # check if top guess is correct
        top_guess = group.guess.tolist()[0]
        results.append(int(top_guess == answer))

        # # get the current input vector
        # curr_vec = group.score.tolist()
        # curr_dict = {x.guess: x.score for x in group.itertuples()}
        # diff_vec, isnew_vec = [], []
        # for i, x in enumerate(group.itertuples()):
        #     if x.guess not in prev_dict:
        #         diff_vec.append(x.score)
        #         isnew_vec.append(1)
        #     else:
        #         diff_vec.append(x.score - prev_dict[x.guess])
        #         isnew_vec.append(0)
        # vec = curr_vec + prev_vec + diff_vec + isnew_vec
        # assert(len(vec) == 4 * NUM_GUESSES)
        # prev_vec = curr_vec
        # prev_dict = curr_dict

        vec = {x.guess: x.score for x in group.itertuples()}

        guess_vecs.append(vec)

    queue.put(qnum)
    return (qnum, answer_id, guess_vecs, results)

def load_quizbowl(): 
    log.info('Loading data')
    question_db = QuestionDatabase()
    quizbowl_db = QuizBowlDataset(MIN_ANSWERS)
    all_questions = question_db.all_questions()
    if not os.path.isfile(OPTIONS_DIR):
        log.info('Loading the set of options')
        all_guesses = AbstractGuesser.load_guesses(GUESSES_DIR, folds=c.ALL_FOLDS)
        all_options = set(all_guesses.guess)

        pool = Pool(8)
        folds = quizbowl_db.questions_by_fold()
        all_options.update({format_guess(q.page) for q in folds['train'].values()})
        all_options.update({format_guess(q.page) for q in folds['dev'].values()})

        id2option = list(all_options)
        pickle.dump(id2option, open(OPTIONS_DIR, 'wb'))
    else:
        id2option = pickle.load(open(OPTIONS_DIR, 'rb'))
    option2id = {o: i for i, o in enumerate(id2option)}
    num_options = len(id2option)
    log.info('Number of options', len(id2option))

    all_guesses = dict()
    # folds = c.ALL_FOLDS
    folds = ['test', 'dev']
    for fold in folds:
        save_dir = '%s_processed.pickle' % (GUESSES_DIR + fold)
        if os.path.isfile(save_dir):
            all_guesses[fold] = pickle.load(open(save_dir, 'rb'))
            log.info('Loading {0} guesses'.format(fold))
            continue

        log.info('Processing {0} guesses'.format(fold))
        guesses = AbstractGuesser.load_guesses(GUESSES_DIR, folds=[fold])
        pool = Pool(16)
        manager = Manager()
        queue = manager.Queue()
        worker = partial(_process_question_df, option2id, all_questions)
        inputs = [(question, queue) for question in guesses.groupby('qnum')]
        total_size = len(inputs)
        result = pool.map_async(worker, inputs)

        # monitor loop
        while True:
            if result.ready():
                break
            else:
                size = queue.qsize()
                sys.stderr.write('\r[df data] done: {0}/{1}'.format(size, total_size))
                time.sleep(0.1)

        log.info('Processed {0} guesses saved to '.format(fold, save_dir))
        all_guesses[fold] = result.get()
        pickle.dump(all_guesses[fold], open(save_dir, 'wb'))
    return id2option, all_guesses

def metric(prediction, ground_truth, mask):
    assert prediction.shape == ground_truth.shape
    assert prediction.shape == mask.shape
    stats = dict()
    match = ((prediction == ground_truth) * mask)
    positive = (ground_truth * mask)
    positive_match = (match * positive).sum()
    total = mask.sum()
    stats['acc'] = (match.sum() / total).tolist()
    stats['pos_acc'] = (positive_match / total).tolist()
    return stats
