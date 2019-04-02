"""
This code produces the plots for our JMLR paper. It assumes you have an anaconda environment like in environment.yaml
Some of this code is copy pasted from the code in qanta/ so that the plotting code can be run independently.
We also include a command for downloading processed results which are non-trivial to reproduce in a single script.
For example, in our analysis we use Stanford CoreNLP in server mode, and that analysis consumes over 60GB of RAM.
Instead of including only the source data we provide intermediate output so that changing plots without rerunning is easy.
The code for all the analysis is provided, but will not run unless a flag to not use cached intermediate results is passed.
"""
import json
import sys
import csv
from collections import defaultdict, Counter
from functional import seq, pseq
import spacy
import unidecode
import nltk
import numpy as np
import pandas as pd
import json
from os import path, makedirs
import requests
import re
from bs4 import BeautifulSoup
from pandas.api.types import CategoricalDtype
from tqdm import tqdm
import click
from pyspark import SparkConf, SparkContext
from plotnine import (
    ggplot, aes,
    xlab, ylab, labs, lims, ggtitle,
    facet_grid, facet_wrap,
    geom_histogram, geom_density, geom_segment, geom_text, geom_bar, geom_violin, geom_boxplot, geom_step, geom_vline,
    geom_line, geom_point, geom_dotplot,
    stat_ecdf, stat_ydensity, stat_bin,
    scale_color_manual, scale_color_discrete,
    scale_fill_manual, scale_fill_discrete,
    scale_x_continuous, scale_y_continuous,
    scale_x_log10, scale_y_log10,
    coord_flip,
    theme, theme_light,
    element_line, element_rect, element_text, element_blank
)

COLORS = [
    '#49afcd', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]
output_path = 'output/plots/'
S3 = 'https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/{}'
FILES = [
    (S3.format('paper/emnlp_2012_questions.csv'), 'data/external/emnlp_2012_questions.csv'),
    (S3.format('paper/emnlp_2014_questions.jsonl'), 'data/external/emnlp_2014_questions.jsonl'),
    (S3.format('qanta.mapped.2018.04.18.json'), 'data/external/datasets/qanta.mapped.2018.04.18.json'),
    (S3.format('paper/syntactic_diversity_table.json', 'data/external/syntactic_diversity_table.json'))
]


def create_spark_context() -> SparkContext:
    spark_conf = SparkConf()\
        .set('spark.rpc.message.maxSize', 300)\
        .setAppName("JMLR")
    return SparkContext.getOrCreate(spark_conf)


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


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def format_question(q):
    q['n_sentences'] = len(q['tokenizations'])
    if q['subcategory'] == 'None':
        q['subcategory'] = None
    q['sentences'] = [q['text'][start:end] for start, end in q['tokenizations']]
    return q

# Each spark worker needs to load its own copy of the NLP model
# separately since its not serializable and thus not broadcastable
nlp_ref = []
AVG_WORD_LENGTH = 5
MIN_WORDS = 12
MIN_CHAR_LENGTH = AVG_WORD_LENGTH * MIN_WORDS


def nlp(text):
    if len(nlp_ref) == 0:
        nlp_ref.append(spacy.load('en_core_web_lg'))

    if len(nlp_ref) == 1:
        decoded_text = unidecode.unidecode(text)
        if len(decoded_text) != len(text):
            eprint('Text must have the same length, falling back to normal text')
            doc = nlp_ref[0](text)
        else:
            doc = nlp_ref[0](decoded_text)
        tokenizations = [(s.start_char, s.end_char) for s in doc.sents]
        first_end_pos = None
        if len(tokenizations) == 0:
            raise ValueError('Zero length question with respect to sentences not allowed')

        for start, end in tokenizations:
            if end < MIN_CHAR_LENGTH:
                continue
            else:
                first_end_pos = end
                break

        if first_end_pos is None:
            first_end_pos = tokenizations[-1][1]

        final_tokenizations = [(0, first_end_pos)]
        for start, end in tokenizations:
            if end <= first_end_pos:
                continue
            else:
                final_tokenizations.append((start, end))

        return final_tokenizations
    else:
        raise ValueError('There should be exactly one nlp model per spark worker')


@click.group(chain=True)
def cli():
    makedirs(output_path, exist_ok=True)


@cli.command()
def qanta_2012_stats():
    """
    This computes and prints dataset statistics for prior versions from EMNLP 2012.
    Published results use private NAQT data, these stats are computed using only public data.
    Use nltk for word tokenization to be consistent with prior analysis.
    Use spacy for sentence tokenization to be consistent with qanta dataset preprocessing.
    (We don't use word tokenizations in dataset preprocessing, we consider it a model detail.)
    """
    with open('data/external/emnlp_2012_questions.csv') as f:
        questions_2012 = list(csv.reader(f))
    
    eprint('N EMNLP 2012 Questions', len(questions_2012))
    questions_2012 = [q[4] for q in questions_2012]
    tokenized_2012 = pseq(questions_2012).map(nltk.word_tokenize).list()
    n_tokens_2012 = sum(len(q) for q in tokenized_2012)
    eprint('N EMNLP 2012 Tokens', n_tokens_2012)
    n_sentences = [len(nlp(q)) for q in tqdm(questions_2012)]
    eprint('N EMNLP 2012 Sentences', sum(n_sentences))


@cli.command()
def qanta_2014_stats():
    """
    This computes and prints dataset statistics for prior versions from EMNLP 2014.
    Published results use private NAQT data, these stats are computed using only public data.
    Use nltk for word tokenization to be consistent with prior analysis.
    Use spacy for sentence tokenization to be consistent with qanta dataset preprocessing.
    (We don't use word tokenizations in dataset preprocessing, we consider it a model detail.)
    """
    questions_2014 = pseq.jsonl('data/external/emnlp_2014_questions.jsonl').cache()
    eprint('N EMNLP 2014 Questions', questions_2014.len())
    n_tokens_2014 = questions_2014.map(lambda q: q['question']).map(nltk.word_tokenize).map(len).sum()
    eprint('N EMNLP 2014 Tokens', n_tokens_2014)
    n_sentences = [len(nlp(q['question'])) for q in tqdm(questions_2014.list())]
    eprint('N EMNLP 2014 Sentences', sum(n_sentences))


@cli.command()
def yoy_growth():
    """
    This creates figures showing the number of questions versus year in dataset
    """
    with open('data/external/datasets/qanta.mapped.2018.04.18.json') as f:
        year_pages = defaultdict(set)
        year_questions = Counter()
        for q in json.load(f)['questions']:
            if q['page'] is not None:
                year_pages[q['year']].add(q['page'])
                year_questions[q['year']] += 1
    start_year = min(year_pages)
    # 2017 is the earlier year we have a full year's worth of data, including partial 2018 isn't accurate
    end_year = min(2017, max(year_pages))
    upto_year_pages = defaultdict(set)
    upto_year_questions = Counter()
    for upto_y in range(start_year, end_year + 1):
        for curr_y in range(start_year, upto_y + 1):
            upto_year_questions[upto_y] += year_questions[curr_y]
            for page in year_pages[curr_y]:
                upto_year_pages[upto_y].add(page)
    year_page_counts = {}
    for y, pages in upto_year_pages.items():
        year_page_counts[y] = len(pages)
    year_page_counts
    year_rows = []
    for y, page_count in year_page_counts.items():
        year_rows.append({'year': y, 'value': page_count, 'Quantity': 'Distinct Answers'})
        year_rows.append({'year': y, 'Quantity': 'Total Questions', 'value': upto_year_questions[y]})
    year_df = pd.DataFrame(year_rows)
    count_cat = CategoricalDtype(categories=['Total Questions', 'Distinct Answers'], ordered=True)
    year_df['Quantity'] = year_df['Quantity'].astype(count_cat)
    eprint(year_df[year_df.Quantity == 'Total Questions'])
    p = (
        ggplot(year_df)
        + aes(x='year', y='value', color='Quantity')
        + geom_line() + geom_point()
        + xlab('Year')
        + ylab('Count up to Year (inclusive)')
        + theme_fs()
        + scale_x_continuous(breaks=list(range(start_year, end_year + 1, 2)))
    )
    p.save(path.join(output_path, 'question_answer_counts.pdf'))


@cli.command()
def syntactic_diversity_plots():
    with open('data/external/syntactic_diversity_table.json') as f:
        rows = json.load(f)
    parse_df = pd.DataFrame(rows)
    parse_df['parse_ratio'] = parse_df['unique_parses'] / parse_df['parses']
    melt_df = pd.melt(
        parse_df,
        id_vars=['dataset', 'depth', 'overlap', 'parses'],
        value_vars=['parse_ratio', 'unique_parses'],
        var_name='metric',
        value_name='y'
    )

    def label_facet(name):
        if name == 'parse_ratio':
            return 'Average Unique Parses per Instance'
        elif name == 'unique_parses':
            return 'Count of Unique Parses'

    def label_y(ys):
        formatted_ys = []
        for y in ys:
            y = str(y)
            if y.endswith('000.0'):
                formatted_ys.append(y[:-5] + 'K')
            else:
                formatted_ys.append(y)
        return formatted_ys
    p = (
    ggplot(melt_df)
        + aes(x='depth', y='y', color='dataset')
        + facet_wrap('metric', scales='free_y', nrow=2, labeller=label_facet)
        + geom_line() + geom_point()
        + xlab('Parse Truncation Depth') + ylab('')
        + scale_color_discrete(name='Dataset')
        + scale_y_continuous(labels=label_y)
        + scale_x_continuous(
            breaks=list(range(1, 11)),
            minor_breaks=list(range(1, 11)),
            limits=[1, 10])
        + theme_fs()
    )
    p.save(path.join(output_path, 'syn_div_plot.pdf'))
    p = (
    ggplot(parse_df)
        + aes(x='depth', y='unique_parses', color='dataset')
        + geom_line() + geom_point()
        + xlab('Parse Truncation Depth')
        + ylab('Count of Unique Parses')
        + scale_color_discrete(name='Dataset')
        + scale_x_continuous(
            breaks=list(range(1, 11)),
            minor_breaks=list(range(1, 11)),
            limits=[1, 10])
        + theme_fs()
    )
    p.save(path.join(output_path, 'n_unique_parses.pdf'))
    p = (
        ggplot(parse_df)
        + aes(x='depth', y='parse_ratio', color='dataset')
        + geom_line() + geom_point()
        + xlab('Parse Truncation Depth')
        + ylab('Average Unique Parses per Instance')
        + scale_color_discrete(name='Dataset')
        + scale_x_continuous(breaks=list(range(1, 11)), minor_breaks=list(range(1, 11)), limits=[1, 10])
        + scale_y_continuous(limits=[0, 1])
        + theme_fs()
    )
    p.save(path.join(output_path, 'parse_ratio.pdf'))


@cli.command()
def download():
    raise NotImplementedError()


if __name__ == '__main__':
    cli()
