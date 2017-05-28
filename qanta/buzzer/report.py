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

from qanta.reporting.new_performance import get_protobowl, _get_top_guesses

def report(buzzes_dir):
    all_questions = QuestionDatabase().all_questions()
    answers = {k: v.page for k, v in all_questions.items()}
    question_texts = {k: v.text for k, v in all_questions.items()}
    protobowl_ids = {k: all_questions[k].protobowl 
                for k in all_questions if all_questions[k].protobowl != ''}
    protobowl_df, user_count = load_protobowl()
    guesses_df = AbstractGuesser.load_guesses(bc.GUESSES_DIR,
            folds=[c.BUZZER_DEV_FOLD])
    questions = guesses_df.groupby('qnum')
    top_guesses = _multiprocess(_get_top_guesses, questions, info='Top guesses', multi=True)
    top_guesses = {k: v for k, v in top_guesses}
    
    with open(buzzes_dir, 'rb') as infile:
        buzzes = pickle.load(infile)
            
    save_dir = 'output/summary/new_performance/'
    inputs = [top_guesses, buzzes, answers, None, c.BUZZER_DEV_FOLD, save_dir]
    user_answers_thresholds = [1,10,50,100,500,1000,2000]
    threshold_stats = []
    for threshold in user_answers_thresholds:
        pdf1 = protobowl_df[protobowl_df.user_answers > threshold]
        p_inputs = [question_texts, protobowl_ids, pdf1.groupby('qid'), questions] + inputs
        pstats = get_protobowl(p_inputs)
        threshold_stats.append(pstats)
        print(threshold, pstats)
    with open(buzzes_dir + '.pstats', 'wb') as f:
        pickle.dump(threshold_stats, f)
    print([x['reward'] for x in threshold_stats])

if __name__ == '__main__':
    buzzes_dir = 'output/buzzer/neo/dev_buzzes.neo_1.pkl'
    report(buzzes_dir)
