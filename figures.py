#!/usr/bin/env python
import os
import glob
import pandas as pd
import click
import pickle
from plotnine import (
    ggplot, aes, facet_wrap,
    geom_smooth, geom_density, geom_histogram,
    coord_flip
)


QB_ROOT = os.environ.get('QB_ROOT', '')
REPORT_PATTERN = os.path.join(QB_ROOT, 'output/guesser/**/0/guesser_report.pickle')
TMP_NAME_MAP = {
    'ElasticSearchGuesser': 'qanta.guesser.elasticsearch.ElasticSearchGuesser',
    'DanGuesser': 'qanta.guesser.dan.DanGuesser'
}


@click.group()
def main():
    pass


def safe_path(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def category_jmlr(cat):
    if cat in {'Religion', 'Myth', 'Philosophy'}:
        return 'Religion/Myth/Philosophy'
    elif cat == 'Trash':
        return 'Popular Culture'
    else:
        return cat


def int_to_correct(num):
    if num == 1:
        return 'Correct'
    else:
        return 'Wrong'


def save_plot(output_dir, guesser_name, name, plot):
    plot.save(safe_path(os.path.join(output_dir, name)))


class GuesserReport:
    def __init__(self, unpickled_report):
        self.char_df = unpickled_report['char_df']
        self.first_df = unpickled_report['first_df']
        self.full_df = unpickled_report['full_df']
        guesser_name = unpickled_report['guesser_name']
        if guesser_name in TMP_NAME_MAP:
            self.guesser_name = TMP_NAME_MAP[guesser_name]
        else:
            self.guesser_name = guesser_name

        self.full_df['seen'] = 'Full Question'
        self.first_df['seen'] = 'First Sentence'
        self.combined_df = pd.concat([self.full_df, self.first_df])
        self.combined_df['Outcome'] = self.combined_df.correct.map(int_to_correct)
        self.combined_df['category_jmlr'] = self.combined_df.category.map(category_jmlr)
        self.combined_df = self.combined_df.groupby(['qanta_id', 'seen']).nth(0).reset_index()

        self.char_plot_df = self.char_df\
            .sort_values('score', ascending=False)\
            .groupby(['qanta_id', 'char_index'])\
            .nth(0).reset_index()
        self.char_plot_df['category_jmlr'] = self.char_plot_df.category.map(category_jmlr)
        self.char_plot_df['Outcome'] = self.char_plot_df.correct.map(int_to_correct)

    def plot_n_train_vs_accuracy(self):
        return (
            ggplot(self.combined_df) + facet_wrap('seen')
            + aes(x='n_train', fill='Outcome')
            + geom_histogram(binwidth=1)
        )

    def plot_char_percent_vs_accuracy_histogram(self, category=False):
        if category:
            return (
                ggplot(self.char_plot_df) + facet_wrap('category_jmlr')
                + aes(x='char_percent', fill='Outcome')
                + geom_histogram(binwidth=.05)
            )
        else:
            return (
                ggplot(self.char_plot_df)
                + aes(x='char_percent', fill='Outcome')
                + geom_histogram(binwidth=.05)
            )

    def plot_char_percent_vs_accuracy_smooth(self, category=False):
        if category:
            return (
                ggplot(self.char_plot_df)
                + aes(x='char_percent', y='correct', color='category_jmlr')
                + geom_smooth()
            )
        else:
            return (
                    ggplot(self.char_plot_df)
                    + aes(x='char_percent', y='correct')
                    + geom_smooth()
            )




@main.command()
@click.argument('output_dir')
def guesser(output_dir):
    for path in glob.glob(REPORT_PATTERN):
        with open(path, 'rb') as f:
            report = GuesserReport(pickle.load(f))

        save_plot(
            output_dir, report.guesser_name,
            'n_train_vs_accuracy.pdf', report.plot_n_train_vs_accuracy()
        )
        save_plot(
            output_dir, report.guesser_name,
            'char_percent_vs_accuracy_histogram.pdf', report.plot_char_percent_vs_accuracy_histogram(category=False)
        )
        save_plot(
            output_dir, report.guesser_name,
            'char_percent_vs_accuracy_histogram_category.pdf',
            report.plot_char_percent_vs_accuracy_histogram(category=True)
        )
        save_plot(
            output_dir, report.guesser_name,
            'char_percent_vs_accuracy_smooth.pdf', report.plot_char_percent_vs_accuracy_smooth(category=False)
        )
        save_plot(
            output_dir, report.guesser_name,
            'char_percent_vs_accuracy_smooth_category.pdf', report.plot_char_percent_vs_accuracy_smooth(category=True)
        )


if __name__ == '__main__':
    main()