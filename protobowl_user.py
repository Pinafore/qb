import os
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from multiprocessing import Pool
from datetime import datetime
from plotnine import ggplot, aes, theme, geom_density, geom_histogram, \
    geom_point, scale_color_gradient

from qanta.buzzer.util import load_protobowl


def process_user(uid):
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


'''filter users with less than 20 questions
and take the first entry for each question'''
filtered_df_dir = 'filter_20_protobowl.h5'
df, questions = load_protobowl()
if os.path.isfile(filtered_df_dir):
    with pd.HDFStore('filter_20_protobowl.h5') as store:
        df = store['data']
else:
    df = df[df.user_n_records > 20]
    df_grouped = df.groupby('uid')

    uids = list(df_grouped.groups.keys())
    pool = Pool(8)
    user_rows = pool.map(process_user, uids)

    df = pd.DataFrame(list(itertools.chain(*user_rows)), columns=df.columns)
    with pd.HDFStore('filter_20_protobowl.h5') as store:
        store['data'] = df


'''plotting'''
df.result = df.result.apply(lambda x: x is True)
ratio = [p / len(questions[x].split()) for p, x in zip(df.position, df.qid)]
df['ratio'] = pd.Series(ratio, index=df.index)

df_user_grouped = df.groupby('uid')
user_stat = df_user_grouped.agg(np.mean)
log_n_records = np.log(user_stat.user_n_records)
log_n_records = log_n_records.sort_values().values
log_n_records = {'log_n_records': log_n_records, 'index': list(range(len(log_n_records)))}
log_n_records = pd.DataFrame(log_n_records)


user_stat = user_stat.rename(
    index=str,  columns={'result': 'accuracy', 'user_n_records': 'n_records'})
user_stat = user_stat.loc[user_stat.n_records > 20]
print(len(user_stat))
print(len(df.loc[df.user_n_records > 20]))
print(len(df))
print(len(set(df.qid)))

user_stat['log_n_records'] = pd.Series(user_stat.n_records.apply(np.log),
                                       index=user_stat.index)
max_color = user_stat.log_n_records.max()
user_stat['alpha'] = pd.Series(
    user_stat.log_n_records.apply(lambda x: x / max_color), index=user_stat.index)


p0 = ggplot(user_stat) \
        + geom_point(aes(x='ratio', y='accuracy',
                     size='n_records', color='log_n_records', alpha='alpha'),
                     show_legend={'color': False, 'alpha': False, 'size': False}) \
        + scale_color_gradient(high='#e31a1c', low='#ffffcc') \
        + theme(aspect_ratio=1)
p0.save('protobowl_users.pdf')
# p0.draw()
print('p0 done')


p1 = ggplot(user_stat, aes(x='log_n_records', y='..density..')) \
        + geom_histogram(color='#e6550d', fill='#fee6ce') \
        + geom_density() \
        + theme(aspect_ratio=0.3)
p1.save('protobowl_hist.pdf')
# p1.draw()
print('p1 done')


p2 = ggplot(user_stat, aes(x='accuracy', y='..density..')) \
        + geom_histogram(color='#31a354', fill='#e5f5e0') \
        + geom_density(aes(x='accuracy')) \
        + theme(aspect_ratio=0.3)
p2.save('protobowl_acc.pdf')
# p2.draw()
print('p2 done')


p3 = ggplot(user_stat, aes(x='ratio', y='..density..')) \
        + geom_histogram(color='#3182bd', fill='#deebf7') \
        + geom_density(aes(x='ratio')) \
        + theme(aspect_ratio=0.3)
p3.save('protobowl_pos.pdf')
# p3.draw()
print('p3 done')
