import pickle
import argparse
import numpy as np
from itertools import cycle
from collections import defaultdict
from functools import partial
from typing import List, Dict, Tuple

from qanta.guesser.abstract import AbstractGuesser
from qanta.datasets.quiz_bowl import QuestionDatabase
from qanta.util import constants as c
from qanta.buzzer import constants as bc
from qanta import logging
from qanta.buzzer.util import GUESSERS
from qanta.reporting.report_generator import ReportGenerator
from qanta.util.multiprocess import _multiprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

log = logging.get(__name__)
N_GUESSERS = len(GUESSERS)
MAXINT = 99999
HISTO_RATIOS = [0, 0.25, 0.5, 0.75, 1.0]

# continuous valued statistics
EOP_STAT_KEYS_0 = [
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
EOP_STAT_KEYS_1 = [
        'choose_guesser', # the guesser chosen by the buzzer
        'best_guesser' # the best guesser
        ]

HISTO_KEYS = ['acc', 'buzz']  + \
        ['acc_{}'.format(g) for g in GUESSERS] + \
        ['buzz_{}'.format(g) for g in GUESSERS]

LINE_STYLES = {'acc': '-', 'buzz': '-'}
_STYLES = [':', '--', '-.']
for guesser, style in zip(GUESSERS, cycle(_STYLES)):
    LINE_STYLES['acc_{}'.format(guesser)] = style
    LINE_STYLES['buzz_{}'.format(guesser)] = style

def get_top_guesses(inputs):
    (qnum, question), queue = inputs
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
    if queue is not None:
        queue.put(qnum)
    # transpose top_guesses -> n_guessers * length
    return qnum, list(map(list, zip(*top_guesses)))

def end_of_pipeline(buzzes: Dict[int, List[List[float]]],
                    answers: Dict[int, str], inputs) \
                -> Tuple[int, Dict[str, int]]:
    (qnum, top_guesses), queue = inputs
    buzz = buzzes[qnum]
    answer = answers[qnum]

    # top_guesses: n_guessers * length
    length = len(top_guesses[0])
    if len(buzz) != length:
        raise ValueError("Length of buzzes {0} does not match with \
                guesses {1}".format(len(buzz), length))

    stats = {k: -1 for k in EOP_STAT_KEYS_0 + EOP_STAT_KEYS_1}

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
        stats['choose_hopeful'] = int(correct[chosen] != MAXINT)
        stats['reward'] = 10 if pos >= correct[chosen] else -5
        if hopeful:
            stats['choose_best'] = int(chosen == best_guesser)
            stats['late'] = max(0, pos - correct[best_guesser])
            stats['rush'] = max(0, correct[best_guesser] - pos)

    if queue is not None:
        queue.put(qnum)

    return qnum, stats

def histogram(buzzes: Dict[int, List[List[float]]],
              answers: Dict[int, str], inputs) \
            -> Tuple[int, Dict[str, List[int]]]:
    (qnum, top_guesses), queue = inputs
    buzz = buzzes[qnum]
    answer = answers[qnum]

    # top_guesses: n_guessers * length
    length = len(top_guesses[0])
    if len(buzz) != length:
        raise ValueError("Length of buzzes {0} does not match with \
                guesses {1}".format(len(buzz), length))

    # n_guessers * length -> length * n_guessers
    top_guesses = list(map(list, zip(*top_guesses)))
    correct = [[int(x == answer) for x in g] for g in top_guesses]
    stats = {k: [-1 for _ in HISTO_RATIOS] for k in HISTO_KEYS}

    for i, r in enumerate(HISTO_RATIOS):
        pos = int(length * r)
        cor = sum(sum(x) for x in correct[:pos])
        buz = sum(np.argmax(x) < N_GUESSERS for x in buzz[:pos])
        stats['acc'][i] = int(cor > 0)
        stats['buzz'][i] = int(buz > 0)
        for j, g in enumerate(GUESSERS):
            cor = sum(x[j] for x in correct[:pos])
            buz = sum(np.argmax(x) == j for x in buzz[:pos])
            stats['acc_{}'.format(g)][i] = int(cor > 0)
            stats['buzz_{}'.format(g)][i] = int(buz > 0)

    if queue is not None:
        queue.put(qnum)

    return qnum, stats

def generate(buzzes, answers, guesses_df, fold, checkpoint_dir=None,
        plot_dir=None, multiprocessing=True):
    questions = guesses_df.groupby('qnum')

    # qnum -> n_guessers * length
    top_guesses = _multiprocess(get_top_guesses, questions, 
        info='Top guesses', multi=multiprocessing)
    top_guesses = {k: v for k, v in top_guesses}

    ############# end-of-pipeline stats ############# 

    inputs = top_guesses.items()
    worker = partial(end_of_pipeline, buzzes, answers)
    eop_stats = _multiprocess(worker, inputs, info='End-of-pipeline stats',
            multi=multiprocessing)

    # qnum -> key -> int
    eop_stats = {k: v for k, v in eop_stats}
    # key -> int
    _eop_stats = defaultdict(lambda: [])

    eop_output = ""
    for qnum, stat in eop_stats.items():
        for key in EOP_STAT_KEYS_0 + EOP_STAT_KEYS_1:
            if stat[key] != -1:
                _eop_stats[key].append(stat[key])

    for key in EOP_STAT_KEYS_0:
        values = _eop_stats[key]
        value = sum(values) / len(values) if len(values) > 0 else 0
        output = "{0} {1:.3f}".format(key, value)
        eop_output += output + '\n'
        print(output)

    for key in EOP_STAT_KEYS_1:
        output = key
        values = _eop_stats[key]
        for i in range(len(GUESSERS)):
            output += " {0} {1}".format(GUESSERS[i], values.count(i))
        eop_output += output + '\n'
        print(output)

    ############# histogram stats ############# 
    inputs = top_guesses.items()
    worker = partial(histogram, buzzes, answers)
    his_stats = _multiprocess(worker, inputs, info='Histogram stats',
            multi=multiprocessing)
    # qnum -> key -> list(int)
    his_stats = {k: v for k, v in his_stats}
    # key -> list(int)
    _his_stats = defaultdict(lambda: [[] for _ in HISTO_RATIOS])

    for stats in his_stats.values():
        for key in HISTO_KEYS:
            for i, r in enumerate(HISTO_RATIOS):
                if stats[key][i] != -1:
                    _his_stats[key][i].append(stats[key][i])

    for key in HISTO_KEYS:
        for i, r in enumerate(HISTO_RATIOS):
            s = _his_stats[key][i]
            _his_stats[key][i] = sum(s) / len(s) if len(s) > 0 else 0

    _his_stats = dict(_his_stats)
    
    his_output = ""
    for i, r in enumerate(HISTO_RATIOS):
        output = "{}:".format(r)
        for key in HISTO_KEYS:
            output += "  {0} {1:.2f}".format(key, _his_stats[key][i])
        his_output += output + '\n'
        print(output)
	
    if plot_dir is not None:
        lines = []
        for k, v in _his_stats.items():
            lines.append(plt.plot(HISTO_RATIOS, v, LINE_STYLES[k], label=k)[0])
        plt.legend(handles=lines)
        plt.savefig(plot_dir, dpi=200, format='png')
        plt.clf()

    if checkpoint_dir is not None:
        checkpoint = {
                'buzzes': buzzes, 
                'top_guesses': top_guesses,
                'eop_keys': EOP_STAT_KEYS_0 + EOP_STAT_KEYS_1,
                'his_keys': HISTO_KEYS,
                'eop_stats': eop_stats,
                'his_stats': his_stats,
                '_his_stats': _his_stats
                }
        with open(checkpoint_dir, 'wb') as outfile:
            pickle.dump(checkpoint, outfile)

    return eop_output, his_output

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fold', default=None)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args.fold != None:
        folds = [args.fold]
    else:
        folds = c.BUZZ_FOLDS

    all_questions = QuestionDatabase().all_questions()
    answers = {k: v.page for k, v in all_questions.items()}

    variables = dict()
    for fold in folds:
        guesses_df = AbstractGuesser.load_guesses(
                bc.GUESSES_DIR, folds=[fold])

        buzzes_dir = bc.BUZZES_DIR.format(fold)
        with open(buzzes_dir, 'rb') as infile:
            buzzes = pickle.load(infile)
        log.info('Buzzes loaded from {}.'.format(buzzes_dir))

        checkpoint_dir = "output/summary/performance_{}.pkl".format(fold)
        plot_dir = "output/summary/performance_{}_his.png".format(fold)
        eop_output, his_output = generate(buzzes, answers, guesses_df, fold,
                checkpoint_dir, plot_dir)
        variables['eop_{}_output'.format(fold)] = eop_output
        variables['his_{}_output'.format(fold)] = his_output
        variables['his_{}_plot'.format(fold)] = plot_dir

    output = 'output/summary/new_performance.pdf'
    report_generator = ReportGenerator('new_performance.md')
    report_generator.create(variables, output)
