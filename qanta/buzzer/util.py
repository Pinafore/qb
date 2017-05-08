import os
import sys
import time
import pickle
import numpy as np
from collections import namedtuple
from multiprocessing import Pool, Manager
from functools import partial
from typing import List, Dict, Tuple, Optional

from qanta.datasets.quiz_bowl import Question, QuestionDatabase, QuizBowlDataset
from qanta.preprocess import format_guess
from qanta.guesser.abstract import AbstractGuesser
from qanta.util import constants as c
from qanta.util.io import safe_open, safe_path
from qanta.config import conf
from qanta import logging
from qanta.buzzer import constants as bc

log = logging.get(__name__)

def stupid_buzzer(iterator) -> Dict[int, int]:
    '''Buzz by several heuristics.
    '''

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
        reward = ((results * 10) - (hopeful - results) * 5) / hopeful \
                if hopeful > 0 else 0
        num_hopeful += hopeful
        num_results += results
        tot_reward += reward
        for qid, buzz in zip(batch.qids, buzzes):
            qid = qid.tolist()
            buzz_dict[qid] = buzz
    tot_reward /= iterator.size
    log.info('[stupid] {0} {1} {2} {3}'.format(
        num_hopeful, num_results, tot_reward))
    return buzz_dict

def _process_question(option2id: Dict[str, int], 
        all_questions: List[Question], inputs: Tuple) -> \
                Tuple[int, int, List[List[Dict[str, int]]], List[List[int]]]:
    '''Process one question.
    return:
        qnum: question id,
        answer_id: answer id
        guess_dicts: a sequence of guess dictionaries for each guesser
        results: sequence of 0 and 1 for each guesser
    '''
    (qnum, question), queue = inputs

    qnum = int(qnum)
    answer = format_guess(all_questions[qnum].page)
    if answer in option2id:
        answer_id = option2id[answer]
    else:
        answer_id = len(option2id)

    guess_dicts = []
    results = []
    for guesser, guesser_group in question.groupby('guesser', sort=True):
        guess_dicts.append([])
        results.append([])
        for pos, sent_token in guesser_group.groupby(['sentence', 'token'], sort=True):
            # check if top guess is correct
            top_guess = sent_token.sort_values('score',
                    ascending=False).iloc[0].guess
            results[-1].append(int(top_guess == answer))
            vec = {x.guess: x.score for x in sent_token.itertuples()}
            guess_dicts[-1].append(vec)

    queue.put(qnum)
    return qnum, answer_id, guess_dicts, results

def load_quizbowl(folds=['dev', 'test']) -> Tuple[Dict[str, int], Dict[str, list]]: 
    log.info('Loading data')
    question_db = QuestionDatabase()
    quizbowl_db = QuizBowlDataset(bc.MIN_ANSWERS)
    all_questions = question_db.all_questions()
    if not os.path.isfile(bc.OPTIONS_DIR):
        log.info('Loading the set of options')
        dev_guesses = AbstractGuesser.load_guesses(bc.GUESSES_DIR, folds=['dev'])
        all_options = set(dev_guesses.guess)

        by_folds = quizbowl_db.questions_by_fold()
        train_dev_questions = quizbowl_db.questions_in_by_folds(['train', 'dev'])
        all_options.update({format_guess(q.page) for q in train_dev_questions})

        id2option = list(all_options)
        with open(safe_path(bc.OPTIONS_DIR), 'wb') as outfile:
            pickle.dump(id2option, outfile)
    else:
        with open(safe_path(bc.OPTIONS_DIR), 'rb') as infile:
            id2option = pickle.load(infile)
    option2id = {o: i for i, o in enumerate(id2option)}
    num_options = len(id2option)
    log.info('Number of options {0}'.format(len(id2option)))

    guesses_by_fold = dict()
    for fold in folds:
        save_dir = '%s_processed.pickle' % (os.path.join(bc.GUESSES_DIR, fold))
        if os.path.isfile(save_dir):
            with open(safe_path(save_dir), 'rb') as infile:
                guesses_by_fold[fold] = pickle.load(infile)
            log.info('Loading {0} guesses'.format(fold))
            continue

        log.info('Processing {0} guesses'.format(fold))
        guesses = AbstractGuesser.load_guesses(bc.GUESSES_DIR, folds=[fold])

        pool = Pool(conf['buzzer']['n_cores'])
        manager = Manager()
        queue = manager.Queue()
        worker = partial(_process_question, option2id, all_questions)
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
        sys.stderr.write('\n')

        log.info('Processed {0} guesses saved to {1}'.format(fold, save_dir))
        guesses_by_fold[fold] = result.get()
        with open(safe_path(save_dir), 'wb') as outfile:
            pickle.dump(guesses_by_fold[fold], outfile)
    return option2id, guesses_by_fold

if __name__ == "__main__":
    option2id, guesses_by_fold = load_quizbowl(['dev', 'test'])
