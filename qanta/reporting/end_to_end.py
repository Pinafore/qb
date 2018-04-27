import os
import pickle
import numpy as np
from multiprocessing import Pool
from functools import partial

from qanta.util.constants import BUZZER_DEV_FOLD
from qanta.datasets.protobowl import load_protobowl
from qanta.datasets.quiz_bowl import QuizBowlDataset
from qanta.buzzer.util import read_data
from qanta.reporting.curve_score import CurveScore
from qanta.buzzer.util import buzzes_dir


report_dir = 'output/reporting'
if not os.path.isdir(report_dir):
    os.mkdir(report_dir)
curve_score = CurveScore()


def _protobowl_scores(q, labels, buzzes, word_positions, records_grouped):
    '''score against protobowl players with system and oracle buzzer'''
    if q.proto_id not in records_grouped.groups:
        return None, None

    rel_pos_buzz = 1
    rel_pos_oracle = 1
    buzz_result = False
    if True in buzzes:
        idx = buzzes.index(True)
        rel_pos_buzz = word_positions[idx] / word_positions[-1]
        buzz_result = labels[idx]
    if True in labels:
        idx = labels.index(True)
        rel_pos_oracle = word_positions[idx] / word_positions[-1]

    score_buzz = 0
    score_oracle = 0
    records = records_grouped.get_group(q.proto_id)
    for r in records.itertuples():
        if r.relative_position <= rel_pos_buzz:
            score_buzz += -10 if r.result else 5 + 10 * labels[-1]
        else:
            score_buzz += 10 if buzz_result else -15

        if r.relative_position <= rel_pos_oracle:
            score_oracle += -10 if r.result else 5 + 10 * labels[-1]
        else:
            score_oracle += 10
    return score_buzz / len(records), score_oracle / len(records)


def _curve_scores(q, labels, buzzes, word_positions, records_grouped):
    '''weighted accuracy with system and oracle buzzer'''
    score_buzz = None
    score_oracle = None
    if True in buzzes:
        idx = buzzes.index(True)
        rel_pos_buzz = word_positions[idx] / word_positions[-1]
        weight = curve_score.get_weight(rel_pos_buzz)
        score_buzz = labels[idx] * weight
    if True in labels:
        idx = labels.index(True)
        rel_pos_oracle = word_positions[idx] / word_positions[-1]
        weight = curve_score.get_weight(rel_pos_oracle)
        score_oracle = weight
    return score_buzz, score_oracle


def run_all_metrics(guesses, buzzes, record_groups, metrics, q):
    word_positions, buzzer_scores = buzzes[q.qanta_id]
    qid, vectors, labels, word_positions = guesses[q.qanta_id]
    buzzes = [b > a for a, b in buzzer_scores]
    scores = [m(q, labels, buzzes, word_positions, record_groups) for m in metrics]
    return scores


def main():
    fold = BUZZER_DEV_FOLD

    # load questions
    print('loading questions')
    questions = QuizBowlDataset(buzzer_train=True).questions_by_fold()
    questions = questions[fold]

    # load guesser outputs
    print('loading guesser outputs')
    guesses = read_data(fold)
    guesses = {x[0]: x for x in guesses}

    # load buzzer outputs
    print('loading buzzer outputs')
    buzz_dir = os.path.join(buzzes_dir.format(fold))
    with open(buzz_dir, 'rb') as f:
        buzzes = pickle.load(f)

    # load protobowl records
    print('loading protobowl records')
    df, _ = load_protobowl()
    record_groups = df.groupby('qid')

    metrics = [_protobowl_scores, _curve_scores]
    pool = Pool(8)
    worker = partial(run_all_metrics, guesses, buzzes, record_groups, metrics)
    scores = pool.map(worker, questions)

    all_scores = list(map(list, zip(*scores)))

    protobowl_scores = all_scores[0]
    protobowl_scores = list(map(list, zip(*protobowl_scores)))
    protobowl_scores = [[x for x in s if x is not None] for s in protobowl_scores]
    print([np.mean(s) for s in protobowl_scores])

    curve_scores = all_scores[1]
    curve_scores = list(map(list, zip(*curve_scores)))
    curve_scores = [[x for x in s if x is not None] for s in curve_scores]
    print([np.mean(s) for s in curve_scores])


if __name__ == '__main__':
    main()
