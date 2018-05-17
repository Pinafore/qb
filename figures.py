#!/usr/bin/env python
import os

if 'DISPLAY' not in os.environ:
    import matplotlib
    matplotlib.use('agg')

import glob
import pandas as pd
import click
import pickle
from typing import List
import numpy as np
from plotnine import (
    ggplot, aes, facet_wrap,
    geom_smooth, geom_density, geom_histogram, geom_bar, geom_line,
    coord_flip, stat_smooth, scale_y_continuous
)


QB_ROOT = os.environ.get('QB_ROOT', '')
REPORT_PATTERN = os.path.join(QB_ROOT, 'output/guesser/**/0/guesser_report_guessdev.pickle')
EXPO_REPORT_PATTERN = os.path.join(QB_ROOT, 'output/guesser/**/0/guesser_report_expo.pickle')


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
    plot.save(safe_path(os.path.join(output_dir, guesser_name, name)))


class GuesserReport:
    def __init__(self, unpickled_report, fold):
        self.fold = fold
        self.char_df = unpickled_report['char_df']
        self.first_df = unpickled_report['first_df']
        self.full_df = unpickled_report['full_df']
        self.guesser_name = unpickled_report['guesser_name']

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
        self.first_accuracy = unpickled_report['first_accuracy']
        self.full_accuracy = unpickled_report['full_accuracy']
        self.unanswerable_answer_percent = unpickled_report['unanswerable_answer_percent']
        self.unanswerable_question_percent = unpickled_report['unanswerable_question_percent']

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
                    + geom_smooth(method='mavg')
            )


GUESSER_SHORT_NAMES = {
    'qanta.guesser.rnn.RnnGuesser': 'RNN',
    'qanta.guesser.dan.DanGuesser': 'DAN',
    'qanta.guesser.elasticsearch.ElasticSearchGuesser': 'ES'
}


def to_shortname(name):
    if name in GUESSER_SHORT_NAMES:
        return GUESSER_SHORT_NAMES[name]
    else:
        return name


class CompareGuesserReport:
    def __init__(self, reports: List[GuesserReport]):
        self.reports = reports
        char_plot_dfs = []
        acc_rows = []
        for r in self.reports:
            char_plot_dfs.append(r.char_plot_df)
            name = to_shortname(r.guesser_name)
            acc_rows.append((r.fold, name, 'first', r.first_accuracy))
            acc_rows.append((r.fold, name, 'full', r.full_accuracy))
        self.char_plot_df = pd.concat(char_plot_dfs)
        self.char_plot_df['guesser_short'] = self.char_plot_df['guesser'].map(to_shortname)
        self.acc_df = pd.DataFrame.from_records(acc_rows, columns=['fold', 'guesser', 'position', 'accuracy'])

    def plot_char_percent_vs_accuracy_smooth(self, expo=False):
        if expo:
            return (
                ggplot(self.char_plot_df) + facet_wrap('fold')
                + aes(x='char_percent', y='correct', color='guesser_short')
                + stat_smooth(method='mavg', se=False, method_args={'window': 500})
                + scale_y_continuous(breaks=np.linspace(0, 1, 21))
            )
        else:
            return (
                ggplot(self.char_plot_df)
                + aes(x='char_percent', y='correct', color='guesser_short')
                + stat_smooth(method='mavg', se=False, method_args={'window': 500})
                + scale_y_continuous(breaks=np.linspace(0, 1, 21))
            )

    def plot_compare_accuracy(self, expo=False):
        if expo:
            return (
                ggplot(self.acc_df) + facet_wrap('position')
                + aes(x='guesser', y='accuracy', fill='fold')
                + geom_bar(stat='identity', position='dodge')
            )
        else:
            return (
                ggplot(self.acc_df) + facet_wrap('position')
                + aes(x='guesser', y='accuracy')
                + geom_bar(stat='identity')
            )


def save_all_plots(output_dir, report: GuesserReport, expo=False):
    if not expo:
        save_plot(
            output_dir, report.guesser_name,
            'n_train_vs_accuracy.pdf', report.plot_n_train_vs_accuracy()
        )
    save_plot(
        output_dir, report.guesser_name,
        'char_percent_vs_accuracy_histogram.pdf', report.plot_char_percent_vs_accuracy_histogram(category=False)
    )

    if not expo:
        save_plot(
            output_dir, report.guesser_name,
            'char_percent_vs_accuracy_histogram_category.pdf',
            report.plot_char_percent_vs_accuracy_histogram(category=True)
        )
    save_plot(
        output_dir, report.guesser_name,
        'char_percent_vs_accuracy_smooth.pdf', report.plot_char_percent_vs_accuracy_smooth(category=False)
    )

    if not expo:
        save_plot(
            output_dir, report.guesser_name,
            'char_percent_vs_accuracy_smooth_category.pdf', report.plot_char_percent_vs_accuracy_smooth(category=True)
        )


@main.command()
@click.option('--use-test', is_flag=True, default=False)
@click.argument('output_dir')
def guesser(use_test, output_dir):
    dev_reports = []
    for path in glob.glob(REPORT_PATTERN):
        with open(path, 'rb') as f:
            report = GuesserReport(pickle.load(f), 'guessdev')
            dev_reports.append(report)

        save_all_plots(output_dir, report)

    expo_reports = []
    expo_output_dir = safe_path(os.path.join(output_dir, 'expo'))
    for path in glob.glob(EXPO_REPORT_PATTERN):
        with open(path, 'rb') as f:
            report = GuesserReport(pickle.load(f), 'expo')
            expo_reports.append(report)

        save_all_plots(expo_output_dir, report, expo=True)

    compare_report = CompareGuesserReport(dev_reports)
    save_plot(
        output_dir, 'compare', 'position_accuracy.pdf',
        compare_report.plot_compare_accuracy()
    )
    save_plot(
        output_dir, 'compare', 'char_accuracy.pdf',
        compare_report.plot_char_percent_vs_accuracy_smooth()
    )

    if len(expo_reports) > 0:
        compare_report = CompareGuesserReport(dev_reports + expo_reports)
        save_plot(
            output_dir, 'compare', 'expo_position_accuracy.pdf',
            compare_report.plot_compare_accuracy(expo=True)
        )
        save_plot(
            output_dir, 'compare', 'expo_char_accuracy.pdf',
            compare_report.plot_char_percent_vs_accuracy_smooth(expo=True)
        )


if __name__ == '__main__':
    main()
