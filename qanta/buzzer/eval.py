import os
import json
import pickle
import chainer
import pandas as pd
import numpy as np
from tqdm import tqdm
from functools import partial

from qanta.buzzer.nets import RNNBuzzer, MLPBuzzer, LinearBuzzer
from qanta.guesser.abstract import AbstractGuesser
from qanta.datasets.quiz_bowl import QuizBowlDataset
from qanta.datasets.protobowl import load_protobowl
from qanta.buzzer.util import read_data, convert_seq
from qanta.util.constants import BUZZER_DEV_FOLD, BUZZER_TEST_FOLD, \
    GUESSER_TEST_FOLD
from qanta.reporting.curve_score import CurveScore

import matplotlib
matplotlib.use('Agg')
from plotnine import ggplot, aes, geom_area, geom_smooth, geom_col


class ThresholdBuzzer:

    def __init__(self, threshold=0.3):
        self.threshold = threshold
        self.model_name = 'ThresholdBuzzer'
        self.model_dir = 'output/buzzer/ThresholdBuzzer'

    def predict(self, xs, softmax=True):
        preds = []
        for x in xs:
            preds.append([])
            for w in x:
                if w[0].data.tolist() > self.threshold:
                    preds[-1].append([1, 0])
                else:
                    preds[-1].append([0, 1])
            preds[-1] = np.array(preds[-1])
            preds[-1] = np.array(preds[-1])
        return preds


def simulate_game(guesses, buzzes, df, question):
    if question.proto_id not in df.groups:
        return [], []

    optimal_pos = 1.1
    buzzing_pos = 1.1
    guess = 'NULL'
    final_guess = 'NULL'

    char_indices, bs = buzzes[question.qanta_id]
    bs = [x[1] > x[0] for x in bs]
    gs = guesses.get_group(question.qanta_id).groupby('char_index')
    if True in bs:
        char_index = char_indices[bs.index(True)]
        buzzing_pos = char_index / len(question.text)
        guess = gs.get_group(char_index).head(1)['guess'].values[0]

    final_guess = gs.get_group(char_indices[-1]).head(1)['guess'].values[0]
    top_guesses = gs.aggregate(lambda x: x.head(1)).guess.tolist()
    if question.page in top_guesses:
        optimal_pos = top_guesses.index(question.page)
        optimal_pos = char_indices[optimal_pos] / len(question.text)

    # print('guess', guess)
    # print('final_guess', final_guess)
    # print('buzzing_pos', buzzing_pos)
    # print('optimal_pos', optimal_pos)

    possibility = []
    outcome = []
    records = df.get_group(question.proto_id)
    # print('$$$$', len(records))
    for record in records.itertuples():
        if record.result and optimal_pos >= record.relative_position:
            possibility.append(False)
        else:
            possibility.append(True)
        score = 0
        if buzzing_pos < record.relative_position:
            if guess == question.page:
                score = 10
            else:
                score = -15 if record.result else -5
        else:
            if record.result:
                score = -10
            else:
                score = 15 if final_guess == question.page else 5
        outcome.append(score)
        # print(score)
    return possibility, outcome


def get_buzzes(model, fold=BUZZER_DEV_FOLD):
    valid = read_data(fold)
    print('# {} data: {}'.format(fold, len(valid)))
    valid_iter = chainer.iterators.SerialIterator(
            valid, 64, repeat=False, shuffle=False)

    predictions = []
    buzzes = dict()
    for batch in tqdm(valid_iter):
        qids, vectors, labels, positions = list(map(list, zip(*batch)))
        batch = convert_seq(batch, device=0)
        preds = model.predict(batch['xs'], softmax=True)
        preds = [p.tolist() for p in preds]
        predictions.extend(preds)
        for i, qid in enumerate(qids):
            buzzes[qid] = []
            for pos, pred in zip(positions[i], preds[i]):
                buzzes[qid].append((pos, pred))
            buzzes[qid] = list(map(list, zip(*buzzes[qid])))

    buzz_dir = os.path.join(model.model_dir,
                            '{}_buzzes.pkl'.format(fold))
    with open(buzz_dir, 'wb') as f:
        pickle.dump(buzzes, f)
    return buzzes


def protobowl(model, fold=BUZZER_DEV_FOLD):
    buzzes = get_buzzes(model, fold)

    '''eval'''
    guesses_dir = AbstractGuesser.output_path(
        'qanta.guesser.rnn', 'RnnGuesser', 0, '')
    guesses_dir = AbstractGuesser.guess_path(guesses_dir, fold, 'char')
    with open(guesses_dir, 'rb') as f:
        guesses = pickle.load(f)
    guesses = guesses.groupby('qanta_id')

    questions = QuizBowlDataset(buzzer_train=True).questions_by_fold()
    questions = questions[fold]

    df = load_protobowl()
    df = df.groupby('qid')

    worker = partial(simulate_game, guesses, buzzes, df)

    possibility = []
    outcome = []
    for question in tqdm(questions):
        pos, out = worker(question)
        possibility += pos
        outcome += out

    result_df = pd.DataFrame({
        'Possibility': possibility,
        'Outcome': outcome,
    })

    result_dir = os.path.join(
        model.model_dir, '{}_protobowl.pkl'.format(fold))
    with open(result_dir, 'wb') as f:
        pickle.dump(result_df, f)


def ew(model, fold=BUZZER_DEV_FOLD):
    buzzes = get_buzzes(model, fold)

    guesses_dir = AbstractGuesser.output_path(
        'qanta.guesser.rnn', 'RnnGuesser', 0, '')
    guesses_dir = AbstractGuesser.guess_path(guesses_dir, fold, 'char')
    with open(guesses_dir, 'rb') as f:
        guesses = pickle.load(f)
    guesses = guesses.groupby('qanta_id')

    answers = dict()
    for qid, bs in buzzes.items():
        answers[qid] = []
        groups = guesses.get_group(qid).groupby('char_index')
        for char_index, scores in zip(*bs):
            guess = groups.get_group(char_index).head(1)['guess']
            guess = guess.values[0]
            buzz = scores[0] < scores[1]
            answers[qid].append({
                'char_index': char_index,
                'guess': guess,
                'buzz': buzz,
            })

    questions = QuizBowlDataset(buzzer_train=True).questions_by_fold()
    questions = {q.qanta_id: q for q in questions[fold]}

    curve_score = CurveScore()
    ew = []
    ew_opt = []
    for qid, answer in answers.items():
        question = questions[qid]
        q = {'text': question.text, 'page': question.page}
        ew.append(curve_score.score(answer, q))
        ew_opt.append(curve_score.score_optimal(answer, q))
    eval_out = {
        'expected_wins': sum(ew),
        'n_examples': len(ew),
        'expected_wins_optimal': sum(ew_opt),
    }
    print(json.dumps(eval_out))
    return eval_out


if __name__ == '__main__':
    model = LinearBuzzer(n_input=22, n_layers=1, n_hidden=50, n_output=2,
                         dropout=0.4)
    # model = RNNBuzzer(n_input=22, n_layers=1, n_hidden=50, n_output=2,
    #                   dropout=0.4)
    # model = MLPBuzzer(n_input=22, n_layers=1, n_hidden=50, n_output=2,
    #                   dropout=0.4)
    # model_path = os.path.join(model.model_dir, 'buzzer.npz')
    # chainer.serializers.load_npz(model_path, model)
    # chainer.backends.cuda.get_device_from_id(0).use()
    model.to_gpu()

    protobowl(model, BUZZER_DEV_FOLD)
    r1 = ew(model, BUZZER_TEST_FOLD)
    r2 = ew(model, GUESSER_TEST_FOLD)
    print((r1['expected_wins'] + r2['expected_wins']) / (r1['n_examples'] + r2['n_examples']))
    print((r1['expected_wins_optimal'] + r2['expected_wins_optimal']) / (r1['n_examples'] + r2['n_examples']))
