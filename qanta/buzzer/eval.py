import os
import json
import pickle
import chainer
import pandas as pd
from tqdm import tqdm

from qanta.buzzer.nets import RNNBuzzer
from qanta.buzzer.args import args
from qanta.guesser.abstract import AbstractGuesser
from qanta.datasets.quiz_bowl import QuizBowlDataset
from qanta.buzzer.util import read_data, convert_seq, report_dir, buzzes_dir
from qanta.util.constants import BUZZER_DEV_FOLD, BUZZER_TEST_FOLD
from qanta.reporting.curve_score import CurveScore

import matplotlib
matplotlib.use('Agg')
from plotnine import ggplot, aes, geom_area, geom_smooth


def eval(fold=BUZZER_DEV_FOLD):
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

    predictions = []
    buzzes = dict()
    for batch in tqdm(valid_iter):
        qids, vectors, labels, positions = list(map(list, zip(*batch)))
        batch = convert_seq(batch, device=args.gpu)
        preds = model.predict(batch['xs'], softmax=True)
        preds = [p.tolist() for p in preds]
        predictions.extend(preds)
        for i in range(len(qids)):
            buzzes[qids[i]] = []
            for pos, pred in zip(positions[i], preds[i]):
                buzzes[qids[i]].append((pos, pred))
            buzzes[qids[i]] = list(map(list, zip(*buzzes[qids[i]])))

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
        'expected_wins': sum(ew) * 1.0 / len(ew),
        'expected_wins_optimal': sum(ew_opt) * 1.0 / len(ew_opt),
    }
    print(json.dumps(eval_out))

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
    eval(BUZZER_TEST_FOLD)
