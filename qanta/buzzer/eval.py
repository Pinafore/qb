import os
import json
import pickle
import chainer
import pandas as pd
import numpy as np
from tqdm import tqdm

from qanta.buzzer.nets import RNNBuzzer
from qanta.buzzer.args import args
from qanta.guesser.abstract import AbstractGuesser
from qanta.datasets.quiz_bowl import QuizBowlDataset
from qanta.datasets.protobowl import load_protobowl
from qanta.buzzer.util import read_data, convert_seq, report_dir, buzzes_dir
from qanta.util.constants import BUZZER_DEV_FOLD, BUZZER_TEST_FOLD, \
    GUESSER_TEST_FOLD
from qanta.reporting.curve_score import CurveScore

import matplotlib
matplotlib.use('Agg')
from plotnine import ggplot, aes, geom_area, geom_smooth, geom_col


class ThresholdBuzzer:

    def __init__(self, threshold=0.3):
        self.threshold = threshold

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


def protobowl(fold=BUZZER_DEV_FOLD):
    valid = read_data(fold)
    print('# {} data: {}'.format(fold, len(valid)))
    valid_iter = chainer.iterators.SerialIterator(
            valid, args.batch_size, repeat=False, shuffle=False)

    args.n_input = valid[0][1][0].shape[0]
    model = RNNBuzzer(args.n_input, args.n_layers, args.n_hidden,
                      args.n_output, args.dropout)
    chainer.serializers.load_npz(args.model_path, model)
    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    predictions = []
    buzzes = dict()
    for batch in tqdm(valid_iter):
        qids, vectors, labels, positions = list(map(list, zip(*batch)))
        batch = convert_seq(batch, device=args.gpu)
        preds = model.predict(batch['xs'], softmax=True)
        preds = [p.tolist() for p in preds]
        predictions.extend(preds)
        for i, qid in enumerate(qids):
            buzzes[qid] = []
            for pos, pred in zip(positions[i], preds[i]):
                buzzes[qid].append((pos, pred))
            buzzes[qid] = list(map(list, zip(*buzzes[qid])))

    '''eval'''
    output_type = 'char'
    guesser_module = 'qanta.guesser.rnn'
    guesser_class = 'RnnGuesser'
    guesser_config_num = 0
    guesses_dir = AbstractGuesser.output_path(
        guesser_module, guesser_class, guesser_config_num, '')
    guesses_dir = AbstractGuesser.guess_path(guesses_dir, fold, output_type)
    with open(guesses_dir, 'rb') as f:
        guesses = pickle.load(f)
    guesses = guesses.groupby('qanta_id')

    questions = QuizBowlDataset(buzzer_train=True).questions_by_fold()
    questions = questions[fold]

    df = load_protobowl()
    df = df.groupby('qid')

    possibility = []
    outcome = []

    for question in questions:
        if question.proto_id not in df:
            continue

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

        records = df.get_group(question.proto_id)
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

    result_df = pd.DataFrame({
        'Possibility': possibility,
        'Outcome': outcome,
    })

    result_df = result_df.groupby(['Possibility', 'Outcome'])
    result_df = result_df.size().reset_index().rename(columns={0: 'Count'})

    p = (
        ggplot(result_df)
        + geom_col(aes(x='Possibility', y='Count', fill='Outcome'))
    )
    p.save(os.path.join(report_dir, 'protobowl_{}.pdf'.format(fold)))

    with open('output/buzzer/protobowl_result.pkl', 'wb') as f:
        pickle.dump(result_df, f)


def ew(fold=BUZZER_DEV_FOLD):
    if not os.path.isdir(report_dir):
        os.mkdir(report_dir)

    valid = read_data(fold)
    print('# {} data: {}'.format(fold, len(valid)))
    valid_iter = chainer.iterators.SerialIterator(
            valid, args.batch_size, repeat=False, shuffle=False)

    args.n_input = valid[0][1][0].shape[0]
    model = RNNBuzzer(args.n_input, args.n_layers, args.n_hidden,
                      args.n_output, args.dropout)
    chainer.serializers.load_npz(args.model_path, model)
    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    # model = ThresholdBuzzer(0.1)

    predictions = []
    buzzes = dict()
    for batch in tqdm(valid_iter):
        qids, vectors, labels, positions = list(map(list, zip(*batch)))
        batch = convert_seq(batch, device=args.gpu)
        preds = model.predict(batch['xs'], softmax=True)
        preds = [p.tolist() for p in preds]
        predictions.extend(preds)
        for i, qid in enumerate(qids):
            buzzes[qid] = []
            for pos, pred in zip(positions[i], preds[i]):
                buzzes[qid].append((pos, pred))
            buzzes[qid] = list(map(list, zip(*buzzes[qid])))

    buzz_dir = os.path.join(buzzes_dir.format(fold))
    with open(buzz_dir, 'wb') as f:
        pickle.dump(buzzes, f)

    output_type = 'char'
    guesser_module = 'qanta.guesser.rnn'
    guesser_class = 'RnnGuesser'
    guesser_config_num = 0
    guesses_dir = AbstractGuesser.output_path(
        guesser_module, guesser_class, guesser_config_num, '')
    guesses_dir = AbstractGuesser.guess_path(guesses_dir, fold, output_type)
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

    results = dict()
    for example_idx in range(len(valid)):
        qid, vectors, labels, positions = valid[example_idx]
        preds = predictions[example_idx]
        q_len = positions[-1]
        for i, pos in enumerate(positions):
            rel_pos = int(100 * pos / q_len)
            if rel_pos not in results:
                results[rel_pos] = []
            results[rel_pos].append((labels[i], preds[i][1]))

    freq = {'x': [], 'y': [], 'type': []}
    for k, rs in results.items():
        rs, scores = list(map(list, zip(*rs)))
        freq['x'].append(k / 100)
        freq['y'].append(sum(rs) / len(rs))
        freq['type'].append('acc')

        freq['x'].append(k / 100)
        freq['y'].append(sum(x > 0.5 for x in scores) / len(scores))
        freq['type'].append('0.5')

        freq['x'].append(k / 100)
        freq['y'].append(sum(x > 0.3 for x in scores) / len(scores))
        freq['type'].append('0.3')

        freq['x'].append(k / 100)
        freq['y'].append(sum(x > 0.7 for x in scores) / len(scores))
        freq['type'].append('0.7')
    freq_df = pd.DataFrame(freq)

    p0 = ggplot(freq_df) + geom_smooth(aes(x='x', y='y', color='type'))
    p0.save(os.path.join(report_dir, '{}_acc_buzz.pdf'.format(fold)))

    stack_freq = {'x': [], 'y': [], 'type': []}
    threshold = 0.5
    for k, rs in results.items():
        num = len(rs)
        only_oracle = sum((c == 1 and b <= threshold) for c, b in rs)
        only_buzzer = sum((c == 0 and b > threshold) for c, b in rs)
        both = sum((c == 1 and b > threshold) for c, b in rs)
        neither = sum((c == 0 and b <= threshold) for c, b in rs)

        stack_freq['x'].append(k / 100)
        stack_freq['y'].append(only_oracle / num)
        stack_freq['type'].append('only_oracle')

        stack_freq['x'].append(k / 100)
        stack_freq['y'].append(only_buzzer / num)
        stack_freq['type'].append('only_buzzer')

        stack_freq['x'].append(k / 100)
        stack_freq['y'].append(both / num)
        stack_freq['type'].append('both')

        stack_freq['x'].append(k / 100)
        stack_freq['y'].append(neither / num)
        stack_freq['type'].append('neither')

    stack_freq_df = pd.DataFrame(stack_freq)

    p1 = ggplot(stack_freq_df) + geom_area(aes(x='x', y='y', fill='type'))
    p1.save(os.path.join(report_dir, '{}_stack_area.pdf'.format(fold)))


if __name__ == '__main__':
    protobowl(BUZZER_DEV_FOLD)
    # r1 = ew(BUZZER_TEST_FOLD)
    # r2 = ew(GUESSER_TEST_FOLD)
    # print((r1['expected_wins'] + r2['expected_wins']) / (r1['n_examples'] + r2['n_examples']))
    # print((r1['expected_wins_optimal'] + r2['expected_wins_optimal']) / (r1['n_examples'] + r2['n_examples']))
