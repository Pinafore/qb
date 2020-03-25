import os
import sys
import json
import codecs
import pickle
import pathlib
import itertools
import numpy as np
import pandas as pd
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
from plotnine import ggplot, aes, theme, geom_density, geom_histogram, \
    geom_point, scale_color_gradient, labs


def process_log_line(x):
    '''Process a single line of the log'''
    obj = x['object']
    date = datetime.strptime(x['date'][:-6], '%a %b %d %Y %H:%M:%S %Z%z')
    relative_position = obj['time_elapsed'] / obj['time_remaining']
    return [date,
            obj['guess'],
            obj['qid'],
            obj['time_elapsed'],
            obj['time_remaining'],
            relative_position,
            obj['ruling'],
            obj['user']['id']],\
        obj['qid'],\
        obj['question_text']


# remove duplicate records
def remove_duplicate(df_grouped, uid):
    '''For each user, only take the first record for each question'''
    group = df_grouped.get_group(uid)
    user_questions = set()
    index = group.date.sort_values()
    rows = []
    for _, row in group.loc[index.index].iterrows():
        if row.qid in user_questions:
            continue
        user_questions.add(row.qid)
        rows.append(row)
    for j, row in enumerate(rows):
        rows[j].user_n_records = len(rows)
    return rows


def load_protobowl(
        protobowl_dir='data/external/datasets/protobowl/protobowl-042818.log',
        min_user_questions=20,
        get_questions=False):
    '''Parse protobowl log, return buzz data and questions.
    Filter users that answered less than `min_user_questions` questions.
    Remove duplicates: for each user, only keep the first record for each
    question.

    Args
        protobowl_dir: json log
        min_user_questions: minimum number of questions answered
    Return
        df: dataframe of buzzing records
        questions: protobowl questions
    '''
    df_dir = protobowl_dir + '.h5'
    question_dir = protobowl_dir + '.questions.pkl'

    if os.path.exists(df_dir) and os.path.exists(df_dir):
        with pd.HDFStore(df_dir) as store:
            df = store['data']
        with open(question_dir, 'rb') as f:
            questions = pickle.load(f)
        if get_questions:
            return df, questions
        else:
            return df

    # parse protobowl json log
    data = []
    count = 0
    user_questions = defaultdict(set)
    questions = dict()
    with codecs.open(protobowl_dir, 'r', 'utf-8') as f:
        line = f.readline()
        while line is not None:
            line = line.strip()
            if len(line) < 1:
                break
            while not line.endswith('}}'):
                _line = f.readline()
                if _line is None:
                    break
                line += _line.strip()
            try:
                line = json.loads(line)
            except ValueError:
                line = f.readline()
                if line is None:
                    break
                continue
            count += 1
            if count % 10000 == 0:
                sys.stderr.write('\rdone: {}/5130000'.format(count))
            x, qid, question_text = process_log_line(line)
            if qid not in questions:
                questions[qid] = question_text
            user_questions[x[-1]].add(qid)  # x[-1] is uid
            data.append(x)
            line = f.readline()

    # filter users without enough questions
    filtered_data = []
    for x in data:
        uid = x[-1]
        if len(user_questions[uid]) >= min_user_questions:
            x.append(len(user_questions[uid]))
            filtered_data.append(x)

    df = pd.DataFrame(
        filtered_data,
        columns=['date', 'guess', 'qid', 'time_elapsed', 'time_remaining',
                 'relative_position', 'result', 'uid', 'user_n_records'])

    df_grouped = df.groupby('uid')
    uids = list(df_grouped.groups.keys())
    pool = Pool(8)
    _remove_duplicate = partial(remove_duplicate, df_grouped)
    user_rows = pool.map(_remove_duplicate, uids)
    df = pd.DataFrame(list(itertools.chain(*user_rows)), columns=df.columns)
    df_grouped = df.groupby('uid')

    print('{} users'.format(len(df_grouped)))
    print('{} records'.format(len(df)))
    print('{} questions'.format(len(set(df.qid))))

    # save
    with pd.HDFStore(df_dir) as store:
        store['data'] = df
    with open(question_dir, 'wb') as f:
        pickle.dump(questions, f)
    if get_questions:
        return df, questions
    else:
        return df


def plot():
    outdir = 'output/protobowl/'
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)

    df = load_protobowl()
    df.result = df.result.apply(lambda x: x is True)
    df['log_n_records'] = df.user_n_records.apply(np.log)

    df_user_grouped = df.groupby('uid')
    user_stat = df_user_grouped.agg(np.mean)
    print('{} users'.format(len(user_stat)))
    print('{} records'.format(len(df)))
    max_color = user_stat.log_n_records.max()
    user_stat['alpha'] = pd.Series(
        user_stat.log_n_records.apply(lambda x: x / max_color), index=user_stat.index)

    # 2D user plot
    p0 = ggplot(user_stat) \
        + geom_point(aes(x='relative_position', y='result',
                     size='user_n_records', color='log_n_records', alpha='alpha'),
                     show_legend={'color': False, 'alpha': False, 'size': False}) \
        + scale_color_gradient(high='#e31a1c', low='#ffffcc') \
        + labs(x='Average buzzing position', y='Accuracy') \
        + theme(aspect_ratio=1)
    p0.save(os.path.join(outdir, 'protobowl_users.pdf'))
    # p0.draw()
    print('p0 done')

    # histogram of number of records
    p1 = ggplot(user_stat, aes(x='log_n_records', y='..density..')) \
        + geom_histogram(color='#e6550d', fill='#fee6ce') \
        + geom_density() \
        + labs(x='Log number of records', y='Density') \
        + theme(aspect_ratio=0.3)
    p1.save(os.path.join(outdir, 'protobowl_hist.pdf'))
    # p1.draw()
    print('p1 done')

    # histogram of accuracy
    p2 = ggplot(user_stat, aes(x='result', y='..density..')) \
        + geom_histogram(color='#31a354', fill='#e5f5e0') \
        + geom_density() \
        + labs(x='Accuracy', y='Density') \
        + theme(aspect_ratio=0.3)
    p2.save(os.path.join(outdir, 'protobowl_acc.pdf'))
    # p2.draw()
    print('p2 done')

    # histogram of buzzing position
    p3 = ggplot(user_stat, aes(x='relative_position', y='..density..')) \
        + geom_histogram(color='#3182bd', fill='#deebf7') \
        + geom_density() \
        + labs(x='Average buzzing position', y='Density') \
        + theme(aspect_ratio=0.3)
    p3.save(os.path.join(outdir, 'protobowl_pos.pdf'))
    # p3.draw()
    print('p3 done')


if __name__ == '__main__':
    plot()
