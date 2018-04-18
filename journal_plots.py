from qanta.datasets.quiz_bowl import QuestionDatabase
from collections import defaultdict, Counter
from functional import seq, pseq
import nltk
import numpy as np
import pandas as pd
from pathlib import Path
import json


from plotnine import (
    ggplot, aes, facet_grid, facet_wrap,
    geom_histogram, geom_density, geom_segment, geom_text, geom_bar, geom_violin, geom_boxplot, geom_step, geom_vline,
    xlab, ylab, ggtitle,
    scale_color_manual, scale_fill_manual, scale_fill_discrete, scale_y_continuous, scale_color_discrete,
    coord_flip, theme,
    stat_ecdf, stat_ydensity
)
COLORS = [
    '#49afcd', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]

output_path = Path('/home/entilzha/code/pinafore-papers/2018_jmlr_qanta/images/')


db = QuestionDatabase()
question_lookup = db.all_questions()

questions = []
for q in question_lookup.values():
    sentences = list(q.text.values())
	questions.append({
		'page': q.page,
		'sentences': sentences,
		'text': ' '.join(q.text.values()),
		'category': q.category.split(':')[0].replace('_', ' '),
		'fold': q.fold,
		'qnum': q.qnum,
		'tournaments': q.tournaments,
		'n_sentences': len(sentences)
	    })

question_category_lookup = {
    q['page']: q['category'] for q in questions
}

with open('/home/entilzha/data/wikidata-claims_instance-of.jsonl') as f:
    claims = [json.loads(l) for l in f]

properties = defaultdict(set)
country_synonyms = {
    'former country',
    'member state of the United Nations',
    'sovereign state',
    'member state of the European Union',
    'permanent member of the United Nations Security Council',
    'member state of the Council of Europe'
}

for r in claims:
    title = r['title']
    if title is None:
        continue
    else:
        title = title.replace(' ', '_')
        obj = r['object']
        if r['property'] == 'instance of' and obj is not None:
            if 'Wikimedia' in obj:
                continue
            if obj in country_synonyms:
                properties[title].add('country')
            else:
                properties[title].add(obj)

object_properties = Counter()
for r in claims:
    if r['title'] is not None and r['object'] is not None and 'Wikimedia' not in r['object']:
        object_properties[r['object']] += 1


def compute_question_property_counts(answers):
    question_property_counts = Counter()
    for page in answers:
        if page in properties:
            q_props = properties[page]
            if 'human' in q_props:
                if len(q_props) == 1:
                    question_property_counts['human'] += 1
                    continue
                elif 'human biblical character' in q_props:
                    question_property_counts['human biblical character'] += 1
                    continue
                elif 'Catholic saint' in q_props:
                    question_property_counts['Catholic saint'] += 1
            for prop in q_props:
                question_property_counts[prop] += 1
    return question_property_counts

qb_question_property_counts = compute_question_property_counts([q['page'] for q in questions])
qb_question_property_counts.most_common(n=20)


def compute_answer_type_assignments(question_property_counts, answers):
    answer_type_assignments = {}
    for page in answers:
        if page in properties:
            q_props = list(properties[page])
            if len(q_props) == 1:
                answer_type_assignments[page] = q_props[0]
            else:
                prop_counts = [question_property_counts[prop] for prop in q_props]
                props_with_counts = zip(prop_counts, q_props)
                _, most_common_prop = max(props_with_counts)
                answer_type_assignments[page] = most_common_prop
        else:
            answer_type_assignments[page] = 'NOMATCH'
    return answer_type_assignments


qb_answer_type_assignments = compute_answer_type_assignments(qb_question_property_counts, [q['page'] for q in questions])

for i in range(len(questions)):
    q = questions[i]
    q['instance of'] = qb_answer_type_assignments[q['page']]


df = pd.DataFrame(questions)

def sub_topic(cat):
    split = cat.split(':')
    if len(split) == 1:
        return None
    else:
        w = split[1]
        if w == 'Europe' or w == 'European':
            return 'European'
        elif 'Religion' in w or 'Mythology' in w:
            return 'Religion/Mythology'
        elif w == 'Classic' or w == 'Classical':
            return 'Classical'
        elif w == 'Other':
            return None
        elif w == 'None':
            return None
        else:
            return split[1]

df['sub_topic'] = df['category'].map(sub_topic)

mean_len = df.n_sentences.mean()
median_len = df.n_sentences.median()
stats_df = pd.DataFrame([
    {'n': mean_len, ' ': 'Mean # of Sentences'},
    {'n': median_len, ' ': 'Median # of Sentences'}
])
p = (
    ggplot(df)
    + aes(x='n_sentences')
    + geom_histogram(binwidth=1)
    + geom_segment(
        aes(x='n', xend='n', y=0, yend=29000, color=' '),
        stats_df
    )
    + xlab('Number of Sentences in Question')
    + ylab('Count')
    + scale_color_manual(values=COLORS)
)
#p.save(str(output_path / 'n_sentence_histogram.pdf'))
p

p = (
    ggplot(df.dropna())
    + facet_wrap('topic', scales='free')
    + aes(x='sub_topic', fill='topic')
    + geom_bar()
    + xlab('Sub-Category') + ylab('Count') + coord_flip()
    + theme(panel_spacing_x=1.4, panel_spacing_y=.3, figure_size=(10, 5))
    + scale_fill_discrete(name="Category")
)
#p.save(str(output_path / 'subcategories.pdf'))
p

p = (
    ggplot(df)
    + aes(x='topic', fill='topic')
    + geom_bar(show_legend=False)
    + xlab('Category') + ylab('Count') + coord_flip()
    + theme(figure_size=(5, 6))
)
#p.save(str(output_path / 'categories.pdf'))
p

with open('/home/entilzha/data/squad/train-v1.1.json') as f:
    squad_dataset = json.load(f)['data']
squad_questions = []
squad_titles = set()
for page in squad_dataset:
    squad_titles.add(page['title'])
    for cqa in page['paragraphs']:
        for qa in cqa['qas']:
            squad_questions.append(qa['question'])

outlier_percentile = .95

# SQuAD Question Lengths
squad_word_dist = pseq(squad_questions).map(lambda s: len(nltk.word_tokenize(s))).list()
squad_word_dist = np.array(squad_word_dist)
squad_word_df = pd.DataFrame({'n': squad_word_dist})
squad_word_df['source'] = '# Words in Question'
squad_word_df['dataset'] = 'SQuAD'
squad_quantile = squad_word_df['n'].quantile(outlier_percentile)
squad_word_df = squad_word_df[squad_word_df.n <= squad_quantile]

# TriviaQA Question Lengths
with open('/home/entilzha/data/triviaqa-unfiltered/unfiltered-web-train.json') as f:
    tqa_data = json.load(f)['Data']
tqa_questions = []
tqa_answers = []
for q in tqa_data:
    if q['Answer']['Type'] == 'WikipediaEntity':
        tqa_questions.append(q['Question'])
        tqa_answers.append(q['Answer']['MatchedWikiEntityName'].replace(' ', '_'))


tqa_word_dist = pseq(tqa_questions).map(lambda s: len(nltk.word_tokenize(s))).list()
tqa_word_dist = np.array(tqa_word_dist)
tqa_word_df = pd.DataFrame({'n': tqa_word_dist})
tqa_word_df['source'] = '# Words in Question'
tqa_word_df['dataset'] = 'TriviaQA'
tqa_quantile = tqa_word_df['n'].quantile(outlier_percentile)
tqa_word_df = tqa_word_df[tqa_word_df.n <= tqa_quantile]

tqa_question_property_counts = compute_question_property_counts(tqa_answers)
tqa_question_property_counts.most_common(n=20)

tqa_answer_type_assignments = compute_answer_type_assignments(tqa_question_property_counts, tqa_answers)

# SimpleQuestions QuestionLengths
with open('/home/entilzha/data/SimpleQuestions/annotated_fb_data_train.txt') as f:
    sq_questions = []
    sq_answers = []
    for line in f:
        splits = line.split('\t')
        sq_questions.append(splits[3].strip())
        sq_answers.append(splits[2].strip())

sq_word_dist = pseq(sq_questions).map(lambda s: len(nltk.word_tokenize(s))).list()
sq_word_dist = np.array(sq_word_dist)
sq_word_df = pd.DataFrame({'n': sq_word_dist})
sq_word_df['source'] = '# Words in Question'
sq_word_df['dataset'] = 'SimpleQuestions'
sq_quantile = sq_word_df['n'].quantile(outlier_percentile)
sq_word_df = sq_word_df[sq_word_df.n <= sq_quantile]

# jeopardy
with open('/home/entilzha/data/jeopardy/jeopardy_questions.json') as f:
    j_questions = []
    j_answers = []
    j_data = json.load(f)
    for q in j_data:
        j_questions.append(q['question'])
        j_answers.append(q['answer'])

j_word_dist = pseq(j_questions).map(lambda s: len(nltk.word_tokenize(s))).list()
j_word_dist = np.array(j_word_dist)
j_word_df = pd.DataFrame({'n': j_word_dist})
j_word_df['source'] = '# Words in Question'
j_word_df['dataset'] = 'Jeopardy!'
j_quantile = j_word_df['n'].quantile(outlier_percentile)
j_word_df = j_word_df[j_word_df.n <= j_quantile]

# Quiz Bowl Question Lengths
sent_word_dist = pseq(questions).flat_map(lambda q: q['sentences']).map(lambda s: len(nltk.word_tokenize(s))).list()
question_word_dist = pseq(questions).map(lambda q: ' '.join(q['sentences'])).map(lambda s: len(nltk.word_tokenize(s))).list()
sent_word_dist = np.array(sent_word_dist)
question_word_dist = np.array(question_word_dist)
sent_word_df = pd.DataFrame({'n': sent_word_dist})
sent_word_df['source'] = '# Words in Question'
sent_word_df['dataset'] = 'QB (sentence)'
sent_quantile = sent_word_df['n'].quantile(outlier_percentile)
sent_word_df = sent_word_df[sent_word_df.n <= sent_quantile]

question_word_df = pd.DataFrame({'n': question_word_dist})
question_word_df['source'] = '# Words in Question'
question_word_df['dataset'] = 'QB (question)'
question_quantile = question_word_df['n'].quantile(outlier_percentile)
question_word_df = question_word_df[question_word_df.n <= question_quantile]

quiz_bowl_word_df = pd.concat([sent_word_df, question_word_df])

word_df = pd.concat([quiz_bowl_word_df, squad_word_df, tqa_word_df, sq_word_df, j_word_df])

p = (
    ggplot(word_df) + aes(x='dataset', y='n')
    + geom_violin(aes(fill='dataset', color='dataset'), trim=True, show_legend={'color': False})# + geom_boxplot(outlier_shape=None, outlier_alpha=0, width=.1)
    + xlab('Dataset') + ylab('Distribution of Length in Words') + coord_flip()
    + scale_fill_discrete(name='Dataset')
)
#p.save(str(output_path / 'length_dist.pdf'))
p

(
    ggplot(word_df) + facet_wrap('dataset', nrow=1) + aes(x='n')
    + geom_histogram(binwidth=2)
    + xlab('Dataset') + ylab('Distribution of Length in Words') + coord_flip()
)

(
    ggplot() + aes(x='n') + facet_wrap('source', scales='free')
    + geom_histogram(data=sent_word_df, binwidth=2)
    + geom_histogram(data=question_word_df, binwidth=2)
    + theme(panel_spacing_x=.5, figure_size=(8, 3))
    + xlab('Length of Example') + ylab('Count')
)

qb_type_df = df.groupby(['instance of', 'category']).count().reset_index()
qb_type_df['n'] = qb_type_df['qnum']
qb_type_df['dataset'] = 'Quiz Bowl'
#qb_type_df = qb_type_df[qb_type_df['instance of'] != 'NOMATCH']
qb_type_df = qb_type_df.sort_values('n', ascending=False)[['instance of', 'category', 'n', 'dataset']][:30]

tqa_counts = Counter(tqa_answer_type_assignments.values())
tqa_type_df = pd.DataFrame(tqa_counts.most_common(30), columns=['instance of', 'n'])
tqa_type_df['category'] = 'NA'
tqa_type_df['dataset'] = 'TriviaQA'
#tqa_type_df = tqa_type_df[tqa_type_df['instance of'] != 'NOMATCH']

type_df = pd.concat([qb_type_df, tqa_type_df])

ordered_categories = list(type_df.groupby('instance of').sum().reset_index().sort_values('n', ascending=False)['instance of'])
type_df['instance of'] = pd.Categorical(type_df['instance of'], categories=ordered_categories, ordered=True)

p = (
    ggplot(type_df) + facet_wrap('dataset') + aes(x='instance of', y='n', fill='category')
    + geom_bar(stat='identity') + coord_flip()
    + xlab('Wikidata.org "instance of" Value') + ylab('Count')
    + scale_fill_discrete(name="Category") + theme(figure_size=(10, 6))
)
#p.save(str(output_path / 'ans_type_dist.pdf'))
p

p = (
    ggplot(tqa_types_df[tqa_types_df['instance of'] != 'NOMATCH']) + aes(x='instance of', y='n')
    + geom_bar(stat='identity') + coord_flip()
    + xlab('Wikidata.org "instance of" Value') + ylab('Count')
)
#p.save(str(output_path / 'ans_type_dist.pdf'))
p

qb_answers = list(df.page.values)

df, sq_questions, j_questions, tqa_questions, squad_questions

qb_a_counts = Counter(qb_answers)
j_a_counts = Counter(j_answers)
tqa_a_counts = Counter(tqa_answers)
sq_a_counts = Counter(sq_answers)

def create_answer_count_df(counts, name):
    rows = []
    for n in counts.values():
        rows.append({'n': n, 'dataset': name})
    ac_df = pd.DataFrame(rows)
    total = ac_df.n.sum()
    ac_df['p'] = ac_df.n / total
    ac_df = ac_df.sort_values('n', ascending=False)
    ac_df['cdf'] = ac_df.p.cumsum()
    ac_df['x'] = list(range(1, len(ac_df) + 1))
    ac_df['r'] = ac_df['x'] / (len(ac_df) + 1)
    return ac_df

answer_count_df = pd.concat([
    create_answer_count_df(qb_a_counts, 'Quiz Bowl'),
    create_answer_count_df(j_a_counts, 'Jeopardy!'),
    create_answer_count_df(tqa_a_counts, 'TriviaQA'),
    create_answer_count_df(sq_a_counts, 'SimpleQuestions')
])

def percent_scale(breakpoints):
    return [f'{100 * b:.0f}%' for b in breakpoints]

p = (
    ggplot(answer_count_df) + aes(x='r', y='cdf', color='dataset')
    + geom_step()
    + xlab('Number of Unique Answers') + ylab('Percent Coverage')
    + scale_y_continuous(labels=percent_scale, breaks=np.linspace(0, 1, num=11))
)
#p.save(str(output_path / 'unique_answer_coverage.pdf'))
p
