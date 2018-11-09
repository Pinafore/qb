import os
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict
from pandas.api.types import CategoricalDtype
from qanta.util.constants import BUZZER_DEV_FOLD
from qanta.datasets.quiz_bowl import QuizBowlDataset
from qanta.guesser.abstract import AbstractGuesser

import matplotlib
matplotlib.use('Agg')
from plotnine import ggplot, aes, geom_area, geom_smooth, geom_col,\
    facet_grid, coord_flip, theme, scale_fill_brewer, theme_light,\
    element_line, element_rect, element_text, element_blank


class theme_fs(theme_light):
    """
    A theme similar to :class:`theme_linedraw` but with light grey
    lines and axes to direct more attention towards the data.
    Parameters
    ----------
    base_size : int, optional
        Base font size. All text sizes are a scaled versions of
        the base font size. Default is 11.
    base_family : str, optional
        Base font family.
    """

    def __init__(self, base_size=11, base_family='DejaVu Sans'):
        theme_light.__init__(self, base_size, base_family)
        self.add_theme(theme(
            axis_ticks=element_line(color='#DDDDDD', size=0.5),
            panel_border=element_rect(fill='None', color='#838383',
                                      size=1),
            strip_background=element_rect(
                fill='#DDDDDD', color='#838383', size=1),
            strip_text_x=element_text(color='black'),
            strip_text_y=element_text(color='black', angle=-90),
            legend_key=element_blank()
        ), inplace=True)


def protobowl(fold=BUZZER_DEV_FOLD):
    df_rnn = pickle.load(
        open('output/buzzer/RNNBuzzer/{}_protobowl.pkl'.format(fold), 'rb'))
    df_rnn = df_rnn.groupby(['Possibility', 'Outcome'])
    df_rnn = df_rnn.size().reset_index().rename(columns={0: 'Count'})
    df_rnn['Model'] = pd.Series(['RNN' for _ in range(len(df_rnn))], index=df_rnn.index)

    df_mlp = pickle.load(
        open('output/buzzer/MLPBuzzer/{}_protobowl.pkl'.format(fold), 'rb'))
    df_mlp = df_mlp.groupby(['Possibility', 'Outcome'])
    df_mlp = df_mlp.size().reset_index().rename(columns={0: 'Count'})
    df_mlp['Model'] = pd.Series(['MLP' for _ in range(len(df_mlp))], index=df_mlp.index)

    df_thr = pickle.load(
        open('output/buzzer/ThresholdBuzzer/{}_protobowl.pkl'.format(fold), 'rb'))
    df_thr = df_thr.groupby(['Possibility', 'Outcome'])
    df_thr = df_thr.size().reset_index().rename(columns={0: 'Count'})
    df_thr['Model'] = pd.Series(['Threshold' for _ in range(len(df_thr))], index=df_thr.index)

    df = df_rnn.append(df_mlp, ignore_index=True)
    df = df.append(df_thr, ignore_index=True)

    outcome_type = CategoricalDtype(categories=[15, 10, 5, 0, -5, -10, -15])
    df['Outcome'] = df['Outcome'].astype(outcome_type)
    model_type = CategoricalDtype(
        categories=['Threshold', 'MLP', 'RNN'])
    df['Model'] = df['Model'].astype(model_type)

    p = (
        ggplot(df)
        + geom_col(aes(x='Possibility', y='Count', fill='Outcome'))
        + facet_grid('Model ~')
        + coord_flip()
        + theme_fs()
        + theme(aspect_ratio=0.2)
        + scale_fill_brewer(type='div', palette=7)
    )

    figure_dir = os.path.join('output/buzzer/{}_protobowl.pdf'.format(fold))
    p.save(figure_dir)


def what():
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
    p0.save(os.path.join(model.model_dir, '{}_acc_buzz.pdf'.format(fold)))


def stack(model_dir, model_name, fold=BUZZER_DEV_FOLD):
    guesses_dir = AbstractGuesser.output_path(
        'qanta.guesser.rnn', 'RnnGuesser', 0, '')
    guesses_dir = AbstractGuesser.guess_path(guesses_dir, fold, 'char')
    with open(guesses_dir, 'rb') as f:
        guesses = pickle.load(f)
    guesses = guesses.groupby('qanta_id')

    buzzes_dir = os.path.join(model_dir, '{}_buzzes.pkl'.format(fold))
    with open(buzzes_dir, 'rb') as f:
        buzzes = pickle.load(f)

    questions = QuizBowlDataset(buzzer_train=True).questions_by_fold()
    questions = {q.qanta_id: q for q in questions[fold]}

    stack_freq = {'Position': [], 'Buzzing': []}
    count = defaultdict(lambda: 0)
    for qid, (char_indices, scores) in buzzes.items():
        gs = guesses.get_group(qid).groupby('char_index')
        gs = gs.aggregate(lambda x: x.head(1)).to_dict()['guess']
        question = questions[qid]
        q_len = len(question.text)
        for i, char_index in enumerate(char_indices):
            buzz_oracle = gs[char_index] == question.page
            buzz_buzzer = scores[i][1] > scores[i][0]

            only_oracle = buzz_oracle and (not buzz_buzzer)
            only_buzzer = buzz_buzzer and (not buzz_oracle)
            both = buzz_buzzer and buzz_oracle
            neither = (not buzz_buzzer) and (not buzz_oracle)

            rel_position = np.round(char_index / q_len, decimals=1)
            count[rel_position] += 1

            if only_oracle:
                stack_freq['Position'].append(rel_position)
                stack_freq['Buzzing'].append('Only optimal')

            if only_buzzer:
                stack_freq['Position'].append(rel_position)
                stack_freq['Buzzing'].append('Only buzzer')

            if both:
                stack_freq['Position'].append(rel_position)
                stack_freq['Buzzing'].append('Both')

            if neither:
                stack_freq['Position'].append(rel_position)
                stack_freq['Buzzing'].append('Neither')

    df = pd.DataFrame(stack_freq)
    df = df.groupby(['Position', 'Buzzing'])
    df = df.size().reset_index().rename(columns={0: 'Frequency'})
    df['Frequency'] = df.apply(
        lambda row: row['Frequency'] / count[row['Position']],
        axis=1)
    df['Model'] = pd.Series([model_name for _ in range(len(df))])
    stack_dir = os.path.join(model_dir, '{}_stack.pkl'.format(fold))
    with open(stack_dir, 'wb') as f:
        pickle.dump(df, f)

    return df


def all_stack(fold=BUZZER_DEV_FOLD):
    df_rnn = stack('output/buzzer/RNNBuzzer', 'RNN', fold)
    df_mlp = stack('output/buzzer/MLPBuzzer', 'MLP', fold)
    df_thr = stack('output/buzzer/ThresholdBuzzer', 'Threshold', fold)
    df = df_rnn.append(df_mlp, ignore_index=True)
    df = df.append(df_thr, ignore_index=True)
    p = (
        ggplot(df)
        + geom_area(aes(x='Position', y='Frequency', fill='Buzzing'))
        + facet_grid('~ Model')
        + theme_fs()
        + theme(aspect_ratio=1)
        + scale_fill_brewer(type='div', palette=7)
    )
    p.save('output/buzzer/{}_stack.pdf'.format(fold))


if __name__ == '__main__':
    all_stack()
