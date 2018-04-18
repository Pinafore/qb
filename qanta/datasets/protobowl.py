import os
import sys
import json
import codecs
import pickle
import itertools
import numpy as np
import pandas as pd
from collections import defaultdict
from multiprocessing import Pool
from datetime import datetime
from plotnine import ggplot, aes, theme, geom_density, geom_histogram, \
    geom_point, scale_color_gradient


def process_log_line(x):
    '''Process a single line of the log'''
    date = datetime.strptime(x['date'][:-6], '%a %b %d %Y %H:%M:%S %Z%z')
    total_time = x['object']['time_elapsed'] + x['object']['time_remaining']
    relative_position = x['object']['time_elapsed'] / total_time
    word_position = int(len(x['object']['question_text'].split()) *
                        relative_position)
    return [date,
            x['object']['guess'],
            x['object']['qid'],
            word_position,
            relative_position,
            x['object']['ruling'],
            x['object']['user']['id']],\
        x['object']['qid'],\
        x['object']['question_text']


def load_protobowl(
        protobowl_dir='data/external/qanta-buzz.log',
        min_user_questions=20):
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
        return df, questions

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
                l = f.readline()
                if l is None:
                    break
                line += l.strip()
            try:
                line = json.loads(line)
            except ValueError:
                line = f.readline()
                if line is None:
                    break
                continue
            count += 1
            if count % 10000 == 0:
                sys.stderr.write('\rdone: {}/9707590'.format(count))
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
        if len(user_questions[uid] >= min_user_questions):
            x.append(len(user_questions[uid]))
            filtered_data.append(x)

    df = pd.DataFrame(
            data, columns=['date', 'guess', 'qid', 'word_position',
                           'relative_position', 'result', 'uid',
                           'user_n_records'])

    # remove duplicate records
    def remove_duplicate(uid):
        '''For each user, only take the first record for each question'''
        group = df_grouped.get_group(uid)
        user_questions = set()
        dates = group.date.apply(lambda x: datetime.strptime(
            x[:-6], '%a %b %d %Y %H:%M:%S %Z%z'))
        index = dates.sort_values()
        rows = []
        for _, row in group.loc[index.index].iterrows():
            if row.qid in user_questions:
                continue
            user_questions.add(row.qid)
            rows.append(row)
        for j, row in enumerate(rows):
            rows[j].user_n_records = len(rows)
        return rows

    df_grouped = df.groupby('uid')
    uids = list(df_grouped.groups.keys())
    pool = Pool(8)
    user_rows = pool.map(remove_duplicate, uids)
    df = pd.DataFrame(list(itertools.chain(*user_rows)), columns=df.columns)

    # save
    with pd.HDFStore(df_dir) as store:
        store['data'] = df
    with open(question_dir, 'wb') as f:
        pickle.dump(questions, f)
    return df, questions


if __name__ == '__main__':
    df, questions = load_protobowl()
