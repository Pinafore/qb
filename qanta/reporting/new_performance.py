import os
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
from qanta.buzzer.util import GUESSERS, load_protobowl
from qanta.reporting.report_generator import ReportGenerator
from qanta.util.multiprocess import _multiprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

log = qlogging.get(__name__)
N_GUESSERS = len(GUESSERS)
MAXINT = 99999
HISTO_RATIOS = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

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

# overall guesser accuracy and buzzing frequency
HISTO_KEYS_0 = ['acc', 'buzz']
#         ['acc_{}'.format(g) for g in GUESSERS] + \
#         ['buzz_{}'.format(g) for g in GUESSERS]
# 
HISTO_KEYS_1 = [
        'buzz_correct',
        'wait_correct',
        'wait_wrong',
        'buzz_wrong',
        'wait_impossible',
        'buzz_impossible',
        'buzz_miss']

LINE_STYLES = {'acc': '-', 'buzz': '-'}
_STYLES = [':', '--', '-.']
for guesser, style in zip(GUESSERS, cycle(_STYLES)):
    LINE_STYLES['acc_{}'.format(guesser)] = style
    LINE_STYLES['buzz_{}'.format(guesser)] = style

def _get_top_guesses(qnum, question):
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
    return qnum, list(map(list, zip(*top_guesses)))

def _get_eop_stats(buzzes: Dict[int, List[List[float]]],
                    answers: Dict[int, str], qnum, top_guesses) \
                -> Tuple[int, Dict[str, int]]:
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
            # stats['late'] = max(0, pos - correct[best_guesser])
            # stats['rush'] = max(0, correct[best_guesser] - pos)
            stats['late'] = int(pos > correct[best_guesser])
            stats['rush'] = int(correct[best_guesser] > pos)

    return qnum, stats

def _get_his_stats(buzzes: Dict[int, List[List[float]]],
              answers: Dict[int, str], qnum, top_guesses) \
            -> Tuple[int, Dict[str, List[int]]]:
    buzz = buzzes[qnum]
    answer = answers[qnum]

    # top_guesses: n_guessers * length
    length = len(top_guesses[0])
    if len(buzz) != length:
        raise ValueError("Length of buzzes {0} does not match with \
                guesses {1}".format(len(buzz), length))

    # n_guessers * length -> length * n_guessers
    top_guesses = list(map(list, zip(*top_guesses)))
    guesser_correct = [[int(x == answer) for x in g] for g in top_guesses]

    buzzer_correct = []
    for i, x in enumerate(buzz):
        x = np.argmax(x)
        if x < N_GUESSERS and guesser_correct[i][x]:
            buzzer_correct.append(1)
        else:
            buzzer_correct.append(0)

    stats = {k: [-1 for _ in HISTO_RATIOS] for k in HISTO_KEYS_0 + HISTO_KEYS_1}

    for i, r in enumerate(HISTO_RATIOS):
        pos = int(length * r)
        for j, g in enumerate(GUESSERS):
            cor = sum(x[j] for x in guesser_correct[:pos])
            buz = sum(np.argmax(x) == j for x in buzz[:pos])
            # stats['acc_{}'.format(g)][i] = int(cor > 0)
            # stats['buzz_{}'.format(g)][i] = int(buz > 0)
        cor_before = sum(sum(x) for x in guesser_correct[:pos])
        cor_after = sum(sum(x) for x in guesser_correct[pos:])
        buz = sum(np.argmax(x) < N_GUESSERS for x in buzz[:pos])
        buz_cor = sum(buzzer_correct[:pos])
        stats['acc'][i] = int(cor_before > 0)
        stats['buzz'][i] = int(buz > 0)
        stats['buzz_correct'][i] = int(buz_cor > 0)
        stats['wait_correct'][i] = int(buz == 0 and cor_before == 0 and cor_after > 0)
        stats['wait_impossible'][i] = int(buz == 0 and cor_before == 0 and cor_after == 0)
        stats['wait_wrong'][i] = int(buz == 0 and cor_before > 0)
        stats['buzz_wrong'][i] = int(buz > 0 and cor_before == 0 and cor_after > 0)
        stats['buzz_miss'][i] = int(buz > 0 and cor_before > 0 and buz_cor == 0)
        stats['buzz_impossible'][i] = int(buz > 0 and cor_before == 0 and cor_after == 0)
        ssum = stats['buzz_correct'][i] + stats['wait_correct'][i] + \
               stats['wait_wrong'][i] + stats['buzz_wrong'][i] + \
               stats['wait_impossible'][i] + stats['buzz_impossible'][i] +\
               stats['buzz_miss'][i]
        assert ssum == 1
    return qnum, stats

def get_eop_stats(top_guesses, buzzes, answers, variables, fold, save_dir):
    log.info('[{}] End-of-pipelin reporting'.format(fold))

    inputs = top_guesses.items()
    worker = partial(_get_eop_stats, buzzes, answers)
    eop_stats = _multiprocess(worker, inputs, info='End-of-pipeline stats',
            multi=True)

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
        _eop_stats[key] = value
        output = "{0} {1:.3f}".format(key, value)
        eop_output += output + '\n'
        # print(output)

    for key in EOP_STAT_KEYS_1:
        output = key
        values = _eop_stats[key]
        _eop_stats[key] = dict()
        for i, guesser in enumerate(GUESSERS):
            output += " {0} {1}".format(guesser, values.count(i))
            _eop_stats[key][guesser] = values.count(i)
        eop_output += output + '\n'
        # print(output)

    if variables is not None:
        variables['eop_stats'][fold] = _eop_stats

    return _eop_stats

def get_his_stats(top_guesses, buzzes, answers, variables, fold, save_dir):
    log.info('[{}] Histogram reporting'.format(fold))

    inputs = top_guesses.items()
    worker = partial(_get_his_stats, buzzes, answers)
    his_stats = _multiprocess(worker, inputs, info='Histogram stats',
            multi=True)
    # qnum -> key -> list(int)
    his_stats = {k: v for k, v in his_stats}
    # key -> list(int)
    _his_stats = defaultdict(lambda: [[] for _ in HISTO_RATIOS])

    for stats in his_stats.values():
        for key in HISTO_KEYS_0 + HISTO_KEYS_1:
            for i, r in enumerate(HISTO_RATIOS):
                if stats[key][i] != -1:
                    _his_stats[key][i].append(stats[key][i])

    for key in HISTO_KEYS_0 + HISTO_KEYS_1:
        for i, r in enumerate(HISTO_RATIOS):
            s = _his_stats[key][i]
            _his_stats[key][i] = sum(s) / len(s) if len(s) > 0 else 0

    _his_stats = dict(_his_stats)
    
    his_output = ""
    for i, r in enumerate(HISTO_RATIOS):
        output = "{}:".format(r)
        for key in HISTO_KEYS_0 + HISTO_KEYS_1:
            output += "  {0} {1:.2f}".format(key, _his_stats[key][i])
        his_output += output + '\n'
        # print(output)

    ##### plot lines #####
    fig, ax = plt.subplots()
    lines = []
    for k in HISTO_KEYS_0:
        v = _his_stats[k]
        lines.append(plt.plot(HISTO_RATIOS, v, LINE_STYLES[k], label=k)[0])

    ax.set_xticks(HISTO_RATIOS)
    plt.legend(handles=lines)
    plt.title('{} histogram lines chart'.format(fold))
    if save_dir is not None:
        his_lines_dir = os.path.join(save_dir, 'his_{}_lines.pdf'.format(fold))
        plt.savefig(his_lines_dir, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

    ##### plot stacked area chart #####
    plt.plot([],[],color='c', alpha=0.5, label='buzz_correct')
    plt.plot([],[],color='y', alpha=0.5, label='buzz_miss')
    plt.plot([],[],color='r', alpha=0.5, label='buzz_wrong')
    plt.plot([],[],color='k', alpha=0.5, label='buzz_impossible')
    plt.plot([],[],color='m', alpha=0.5, label='wait_wrong')
    plt.plot([],[],color='g', alpha=0.5, label='wait_correct')
    plt.plot([],[],color='w', alpha=0.5, label='wait_impossible')

    plt.stackplot(list(range(len(HISTO_RATIOS))), 
            _his_stats['buzz_correct'], 
            _his_stats['buzz_miss'],
            _his_stats['buzz_wrong'], 
            _his_stats['buzz_impossible'],
            _his_stats['wait_wrong'], 
            _his_stats['wait_correct'], 
            _his_stats['wait_impossible'], 
            colors=['c', 'y', 'r', 'k', 'm', 'g', 'w'], alpha=0.5)
    plt.legend()
    plt.title('{} stacked area chart'.format(fold))
    if save_dir is not None:
        his_stacked_dir = os.path.join(save_dir, 'his_{}_stacked.pdf'.format(fold))
        plt.savefig(his_stacked_dir, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

    if variables is not None:
        variables['his_stats'][fold] = _his_stats
        variables['his_lines'][fold] = his_lines_dir
        variables['his_stacked'][fold] = his_stacked_dir

    return _his_stats

def get_hyper_search(top_guesses, buzzes, answers, variables, fold, save_dir):
    log.info('[{}] Hyperparameter search reporting'.format(fold))

    cfg_buzzes_dir = 'output/buzzer/cfg_buzzes_{}.pkl'.format(fold)
    if not os.path.exists(cfg_buzzes_dir):
        return

    with open(cfg_buzzes_dir, 'rb') as infile:
        cfg_buzzes = pickle.load(infile)
    n_configs = len(cfg_buzzes)
    
    configs, rushs, lates = [], [], []
    choose_best, choose_hopeful  = [], []
    for config, buzzes in cfg_buzzes:
        s = get_eop_stats(top_guesses, buzzes, answers, None, fold, save_dir)
        configs.append(config)
        rushs.append(s['rush'])
        lates.append(s['late'])
        choose_best.append(s['choose_best'])
        choose_hopeful.append(s['choose_hopeful'])

    config_names = list(range(n_configs))
        
    ##### plot rush and late #####
    pos = list(range(n_configs))
    width = 0.5
    fig, ax = plt.subplots()
    bars = []
    bars.append(plt.bar(pos, rushs, width, alpha=0.5, color='#EE3224')[0])
    bars.append(plt.bar(pos, lates, width, bottom=rushs, alpha=0.5,
        color='#F78F1E')[0])
    plt.legend(bars, ('rush', 'late'))

    ax.set_ylabel('%')
    ax.set_title('Rush and Late')
    ax.set_xticks([p + 1.42 * width for p in pos])
    ax.set_xticklabels(config_names)

    plt.grid()
    plt.title('{} rush & late chart'.format(fold))
    if save_dir is not None:
        rush_late_dir = os.path.join(save_dir, 'rush_late_{}.pdf'.format(fold))
        plt.savefig(rush_late_dir, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

    ##### plot choose best and choose hopeful #####
    pos = list(range(n_configs))
    width = 0.5
    fig, ax = plt.subplots()
    bars1 = []
    bars1.append(plt.bar(pos, choose_best, width, alpha=0.5,
        color='#EE3224')[0])
    bars1.append(plt.bar(pos, choose_hopeful, width, alpha=0.5,
        color='#F78F1E')[0])
    plt.legend(bars1, ('choose_best', 'choose_hopeful'))

    ax.set_ylabel('%')
    ax.set_title('Choose hopeful and best')
    ax.set_xticks([p + 1.42 * width for p in pos])
    ax.set_xticklabels(config_names)

    plt.grid()
    plt.title('{} choices chart'.format(fold))
    if save_dir is not None:
        choice_dir = os.path.join(save_dir, 'choose_{}.pdf'.format(fold))
        plt.savefig(choice_dir, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

    if variables is not None:
        variables['rush_late_plot'][fold] = rush_late_dir
        variables['choice_plot'][fold] = choice_dir
        variables['hype_configs']['dev'] = list(zip(config_names, configs))

def get_protobowl(inputs):
    question_texts, protobowl_ids, protobowl_df, questions, \
            top_guesses, buzzes, answers, variables, fold, save_dir = inputs

    protobowl_keys = ['correct_before', 'correct_after', 
                 'rush_possible', 'rush_impossible', 
                 'late_possible', 'late_impossible',
                 'buzz_before_op', 'buzz_after_op', 'reward']

    avg_stats = {k: [] for k in protobowl_keys}
    n_questions = 0
    for qnum, guess_list in top_guesses.items():
        if qnum not in protobowl_ids:
            continue
        protobowl_id = protobowl_ids[qnum]
        if protobowl_id not in protobowl_df.groups:
            continue
        n_questions += 1

        buzz = buzzes[qnum]
        answer = answers[qnum]

        # position in guesses -> real position
        position_mapping = []
        g_group = questions.get_group(qnum)
        text = question_texts[qnum]
        g_group = g_group.groupby(['sentence', 'token']).groups
        _count = 0
        for sent in text:
            for word, x in enumerate(text[sent].split()):
                if (sent, word) in g_group:
                    position_mapping.append(_count)
                _count += 1
        position_mapping.append(_count)
        if len(position_mapping) != len(buzz):
            print(len(position_mapping), len(buzz))
            continue

        correct_position = len(buzz)
        buzzing_result = False
        for i in range(N_GUESSERS):
            for j in range(len(buzz)):
                if guess_list[i][j] == answer:
                    if j < correct_position:
                        correct_position = j
                        break

        buzzing_position = len(buzz)
        for i in range(len(buzz)):
            choice = np.argmax(buzz[i])
            if choice < N_GUESSERS:
                buzzing_position = position_mapping[i]
                buzzing_result = (guess_list[choice][i] == answer)
                break

        final_choice = np.argmax(buzz[-1][:N_GUESSERS])
        final_result = guess_list[final_choice][-1] == answer

        stats = {k: 0 for k in protobowl_keys}
        n_opponents = 0
        for opponent in protobowl_df.get_group(protobowl_id).itertuples():
            n_opponents += 1
            if opponent.position > buzzing_position:
                stats['buzz_before_op'] += 1
                if buzzing_result:
                    stats['reward'] += 10
                    stats['correct_before'] += 1
                else:
                    stats['reward'] -= 5
                    if correct_position >= opponent.position and opponent.result == True:
                        stats['rush_impossible'] += 1
                    else:
                        stats['rush_possible'] += 1
                    if opponent.result == True:
                        stats['reward'] -= 10
            else:
                stats['buzz_after_op'] += 1
                if opponent.result == True:
                    stats['reward'] -= 10
                    if correct_position <= opponent.position:
                        stats['late_possible'] += 1
                    else:
                        stats['late_impossible'] += 1
                else:
                    stats['reward'] += 5
                    if final_result:
                        stats['correct_after'] += 1
                        stats['reward'] += 10
        for k, v in dict(stats).items():
            avg_stats[k].append(v / n_opponents)
    for k, v in avg_stats.items():
        avg_stats[k] = sum(v) / n_questions

    # plotting
    plot_keys = protobowl_keys[:-1]
    plt.clf()
    ind = 0
    width = 0.5
    labels = []
    for k in plot_keys:
        if k not in avg_stats:
            continue
        plt.bar(ind, avg_stats[k], width)
        labels.append(k)
        ind += width * 2
    plt.xticks(list(range(len(labels))), labels, rotation=30)
    plt.subplots_adjust(bottom=0.3)
    plt.title('{} stats against Protobowl'.format(fold))
    if save_dir is not None:
        plot_dir = os.path.join(save_dir, '{}_protobowl.pdf'.format(fold))
        plt.savefig(plot_dir, bbox_inches='tight')
    else:
        plt.show()
    plt.clf() 

    if variables is not None:
        variables['protobowl_plot'][fold] = plot_dir
        variables['protobowl_stats'][fold] = avg_stats
    return avg_stats

def main(folds, model_name):
    
    all_questions = QuestionDatabase().all_questions()
    answers = {k: v.page for k, v in all_questions.items()}
    question_texts = {k: v.text for k, v in all_questions.items()}
    protobowl_ids = {k: all_questions[k].protobowl 
        for k in all_questions if all_questions[k].protobowl != ''}
    protobowl_df = load_protobowl().groupby('qid')

    save_dir = 'output/summary/new_performance/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # feature -> fold -> value
    variables = defaultdict(lambda: defaultdict())
    for fold in folds:
        guesses_df = AbstractGuesser.load_guesses(
                bc.GUESSES_DIR, folds=[fold])
        questions = guesses_df.groupby('qnum')

        buzzes_dir = bc.BUZZES_DIR.format(fold, model_name)
        with open(buzzes_dir, 'rb') as infile:
            buzzes = pickle.load(infile)
        log.info('Buzzes loaded from {}.'.format(buzzes_dir))

        # qnum -> n_guessers * length
        top_guesses = _multiprocess(_get_top_guesses, questions, 
            info='Top guesses', multi=True)
        top_guesses = {k: v for k, v in top_guesses}
        inputs = [top_guesses, buzzes, answers, variables, fold, save_dir]

        # get_eop_stats(*inputs)
        get_his_stats(*inputs)
        # get_hyper_search(*inputs)
        
        p_inputs = [question_texts, protobowl_ids, protobowl_df, questions] + inputs
        get_protobowl(p_inputs)

    for key, value in variables.items():
        variables[key] = dict(value)
    variables = dict(variables)

    report(variables, save_dir, folds)

def report(variables, save_dir, folds):
    # use this to have jinja skip non-existent features
    jinja_keys = ['his_lines', 'his_stacked', 'rush_late_plot', 'choice_plot',
            'hype_configs', 'protobowl_plot', 'protobowl_stats']
    _variables = {k: dict() for k in jinja_keys}
    _variables.update(variables)
    if len(folds) == 1:
        output = os.path.join(save_dir, 'report_{}.pdf'.format(folds[0]))
    else:
        output = os.path.join(save_dir, 'report_all.pdf')
    report_generator = ReportGenerator('new_performance.md')
    report_generator.create(_variables, output)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fold', default=None)
    parser.add_argument('-m', '--model', required=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args.fold != None:
        folds = [args.fold]
    else:
        folds = c.BUZZER_GENERATION_FOLDS[:-1]
    main(folds, args.model)
