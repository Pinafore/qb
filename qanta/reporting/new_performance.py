import os
import sys
import time
import pickle
import numpy as np
from collections import defaultdict
from functools import partial
from multiprocessing import Pool, Manager
from typing import List, Dict, Tuple, Optional

from qanta.preprocess import format_guess
from qanta.guesser.abstract import AbstractGuesser
from qanta.datasets.quiz_bowl import QuestionDatabase
from qanta.util import constants as c
from qanta.buzzer import constants as bc
from qanta.config import conf
from qanta import logging
from qanta.buzzer.util import GUESSERS

log = logging.get(__name__)
MAXINT = 99999

# continuous valued statistics
STAT_KEYS_0 = [
        'buzz', # did the buzzer buzz
        'choose_best', # did the buzzer choose the best guesser (earliest correct)
        'choose_hopeful', # did the buzzer choose a hopeful guesser
        'rush', # did the buzzer rush (w.r.t to all guessers)
        'late', # did the buzzer buzz too late (w.r.t to all guessers)
        'not_buzzing_when_shouldnt', 
        'reward',
        'hopeful', # is the question hopeful (w.r.t to all guessers)
        'correct' # how many correct buzzers
        ]

# discrete valued statistics
STAT_KEYS_1 = [
        'choose_guesser', # the guesser chosen by the buzzer
        'best_guesser' # the best guesser
        ]

def _examine_question(buzzes: Dict[int, List[List[float]]], 
        answers: Dict[int, str], inputs):
    (qnum, question), queue = inputs
    buzz = buzzes[qnum]
    answer = answers[qnum]
    top_guesses = [] # length * n_guessers
    # FIXME because there can be missing guessers, must iterate position first
    for _, position in question.groupby(['sentence', 'token']):
        top_guesses.append([])
        position = position.groupby('guesser')
        for guesser in GUESSERS:
            if guesser not in position.groups:
                top_guesses[-1].append(None)
            else:
                guesses = position.get_group(guesser).sort_values(
                        'score', ascending=False)
                top_guesses[-1].append(guesses.iloc[0].guess)
    # transpose top_guesses -> n_guessers * length
    top_guesses = list(map(list, zip(*top_guesses)))
    
    length = len(top_guesses[0])
    if len(buzz) != length:
        raise ValueError("Length of buzzes {0} does not match with \
                guesses {1}".format(len(buzz), length))

    stats = {k: -1 for k in STAT_KEYS_0 + STAT_KEYS_1}

    # the first correct position of each guesser
    correct = [g.index(answer) if answer in g else MAXINT for g in top_guesses]
    best_guesser = -1 if np.all(correct == MAXINT) else np.argmin(correct)
    stats['best_guesser'] = best_guesser
    stats['correct'] = sum(x != MAXINT for x in correct)
    stats['hopeful'] = stats['correct'] > 0
    hopeful = stats['hopeful']

    # the buzzing position and chosen guesser
    pos, chosen = -1, -1
    for i in range(length):
        action = np.argmax(buzz[i]) 
        if action < len(GUESSERS):
            pos = i
            chosen = action
            break

    if pos == -1:
        # not buzzing
        stats['buzz'] = 0
        stats['reward'] = 0
        stats['not_buzzing_when_shouldnt'] = int(not hopeful)
    else:
        stats['buzz'] = 1
        stats['choose_guesser'] = chosen
        stats['choose_best'] = int(chosen == best_guesser)
        stats['choose_hopeful'] = int(correct[chosen] != MAXINT)
        stats['reward'] = 10 if pos >= correct[chosen] else -5
        if hopeful:
            stats['late'] = max(0, pos - correct[best_guesser])
            stats['rush'] = max(0, correct[best_guesser] - pos)

    if queue is not None:
        queue.put(qnum)

    return stats

def generate(buzzes, answers, guesses_df, fold):
    questions = guesses_df.groupby('qnum')
    total_size = len(questions)
    stats = defaultdict(lambda: [])

    if True:
        pool = Pool(conf['buzzer']['n_cores'])
        manager = Manager()
        queue = manager.Queue()
        inputs = [(question, queue) for question in questions]
        worker = partial(_examine_question, buzzes, answers)
        result = pool.map_async(worker, inputs)
        # monitor loop
        while True:
            if result.ready():
                break
            else:
                size = queue.qsize()
                sys.stderr.write('\r[performance] done: {0}/{1}'.format(
                    size, total_size))
                time.sleep(0.1)
        sys.stderr.write('\n')
        result = result.get()
        for s in result:
            for k, v in s.items():
                stats[k].append(v)

    else:
        for i, question in enumerate(questions):
            s = _examine_question(buzzes, answers, (question, None))
            for k, v in s.items():
                stats[k].append(v)
            sys.stderr.write('\r[performance] done: {0}/{1}'.format(i, total_size))
        sys.stderr.write('\n')

    all_output = ""
    for key in STAT_KEYS_0:
        vs = [x for x in stats[key] if x != -1]
        v = sum(vs) / len(vs) if len(vs) > 0 else 0
        output = "{0} {1:.3f}".format(key, v)
        all_output += output + '\n'
        log.info(output)

    for key in STAT_KEYS_1:
        vs = [x for x in stats[key] if x != -1]
        output = key
        for i in range(len(GUESSERS)):
            output += " {0} {1}".format(GUESSERS[i], vs.count(i))
        all_output += output + '\n'
        log.info(output)

    return all_output

if __name__ == '__main__':
    fold = 'test'
    buzzes_dir = bc.BUZZES_DIR.format(fold)
    with open(buzzes_dir, 'rb') as infile:
        buzzes = pickle.load(infile)
    log.info('Buzzes loaded from {0}.'.format(buzzes_dir))

    all_questions = QuestionDatabase().all_questions()
    answers = {k: format_guess(v.page) for k, v in all_questions.items()}

    guesses_df = AbstractGuesser.load_guesses(bc.GUESSES_DIR, folds=[fold])
    generate(buzzes, answers, guesses_df, fold)
