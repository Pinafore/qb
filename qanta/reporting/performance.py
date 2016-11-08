import re
from glob import glob
from os import path
import json
import pprint
from collections import namedtuple
from typing import Dict, Set
from enum import Enum
import click
from functional import seq
from functional.pipeline import Sequence
from fn import _

import matplotlib
matplotlib.use('Agg')

from qanta.util.qdb import QuestionDatabase


class Answer(Enum):
    correct = 1
    unanswered_wrong = 2
    unanswered_hopeless_1 = 3
    unanswered_hopeless_classifier = 4
    unanswered_hopeless_dan = 5
    wrong_hopeless_1 = 6
    wrong_hopeless_classifier = 7
    wrong_hopeless_dan = 8
    wrong_early = 9
    wrong_late = 10

ANSWER_PLOT_ORDER = ['correct', 'wrong_late', 'wrong_early', 'unanswered_wrong',
                     'wrong_hopeless_1', 'unanswered_hopeless_1',
                     'wrong_hopeless_classifier', 'unanswered_hopeless_classifier',
                     'wrong_hopeless_dan', 'unanswered_hopeless_dan']

Prediction = namedtuple('Prediction', ['score', 'question', 'sentence', 'token'])
Meta = namedtuple('Meta', ['question', 'sentence', 'token', 'guess'])
Line = namedtuple('Line',
                  ['question', 'sentence', 'token', 'buzz', 'guess', 'answer', 'all_guesses'])
ScoredGuess = namedtuple('ScoredGuess', ['score', 'guess'])

SUMMARY_REGEX = re.compile(r'.*sentence\.([0-9]+)\.json')
ANSWER_REGEX = re.compile(r'.*sentence\.([0-9]+)\.([\-a-z]+)\.answer\.json')


def load_predictions(pred_file: str) -> Sequence:
    def parse_line(line: str) -> Prediction:
        try:
            tokens = line.split()
            score = float(tokens[0])
            if len(tokens) < 2:
                question, sentence, token = None, None, None
            else:
                question, sentence, token = [int(x) for x in tokens[1].split('_')]
            return Prediction(score, question, sentence, token)
        except Exception:
            print("Error parsing line: {0}".format(line))
            raise
    return seq.open(pred_file).map(parse_line)


def load_meta(meta_file: str) -> Sequence:
    def parse_line(line: str) -> Meta:
        tokens = line.split()
        question = int(tokens[0])
        sentence = int(tokens[1])
        token = int(tokens[2])
        guess = ' '.join(tokens[3:])
        return Meta(question, sentence, token, guess)
    return seq.open(meta_file).map(parse_line)


def load_data(pred_file: str, meta_file: str, q_db: QuestionDatabase) -> Sequence:
    preds = load_predictions(pred_file)
    metas = load_meta(meta_file)
    answers = q_db.all_answers()

    def create_line(group):
        question = group[0]
        elements = group[1]
        st_groups = seq(elements).group_by(lambda x: (x[0].sentence, x[0].token)).sorted()
        st_lines = []
        for st, v in st_groups:
            scored_guesses = seq(v)\
                .map(lambda x: ScoredGuess(x[0].score, x[1].guess)).sorted(reverse=True).list()
            st_lines.append(Line(
                question, st[0], st[1],
                scored_guesses[0].score > 0,
                scored_guesses[0].guess, answers[question],
                scored_guesses
            ))
        return question, st_lines

    def fix_missing_label(pm):
        prediction = pm[0]
        meta = pm[1]
        if prediction.question is None or prediction.token is None or prediction.sentence is None:
            print("WARNING: Prediction malformed, fixing with meta line: {0}".format(prediction))
            prediction = Prediction(prediction.score, meta.question, meta.sentence, meta.token)
        assert meta.question == prediction.question
        assert meta.sentence == prediction.sentence
        assert meta.token == prediction.token
        return prediction, meta

    return preds\
        .zip(metas)\
        .map(fix_missing_label)\
        .group_by(lambda x: x[0].question)\
        .map(create_line)


def load_audit(audit_file: str):
    audit_data = {}
    with open(audit_file) as f:
        for line in f:
            qid, evidence = line.split('\t')
            audit_data[qid.strip()] = evidence.strip()
        return audit_data


def compute_answers(data: Sequence, dan_answers: Set[str]):
    questions = {}
    for q, lines in data:
        lines = seq(lines)
        answer = lines.first().answer
        buzz = lines.find(_.buzz)
        if buzz is None:
            if lines.exists(_.guess == answer):
                questions[q] = Answer.unanswered_wrong
            elif answer not in dan_answers:
                questions[q] = Answer.unanswered_hopeless_dan
            else:
                questions[q] = Answer.unanswered_hopeless_1
                if not lines.flat_map(_.all_guesses).exists(_.guess == answer):
                    questions[q] = Answer.unanswered_hopeless_classifier
        elif buzz.guess == buzz.answer:
            questions[q] = Answer.correct
        else:
            correct_buzz = lines.find(_.guess == answer)
            if correct_buzz is None:
                questions[q] = Answer.wrong_hopeless_1
                if answer not in dan_answers:
                    questions[q] = Answer.wrong_hopeless_dan
                else:
                    if not lines.flat_map(_.all_guesses).exists(_.guess == answer):
                        questions[q] = Answer.wrong_hopeless_classifier
            elif (correct_buzz.sentence, correct_buzz.token) < (buzz.sentence, buzz.token):
                questions[q] = Answer.wrong_late
            elif (buzz.sentence, buzz.token) < (correct_buzz.sentence, correct_buzz.token):
                questions[q] = Answer.wrong_early
            else:
                raise ValueError('Unexpected for buzz and correct buzz to be the same')

        if q not in questions:
            raise ValueError('Expected an answer type for question')
    return questions


def compute_statistics(questions: Dict[int, Answer]) -> Sequence:
    n_questions = len(questions)
    empty_set = [(a, 0) for a in Answer]
    results = seq(questions.values())\
        .map(lambda x: (x, 1))
    results = (results + seq(empty_set)).reduce_by_key(lambda x, y: x + y)\
        .map(lambda kv: (str(kv[0]), kv[1] / n_questions if kv[1] > 0 else 0))
    return results


def parse_data(stats_dir):
    def parse_file(file):
        weight = None
        experiment = None
        base_file = path.basename(file)
        m = SUMMARY_REGEX.match(base_file)
        if m:
            weight = int(m.group(1))
            experiment = 'summary'
        m = ANSWER_REGEX.match(base_file)
        if m:
            weight = int(m.group(1))
            experiment = m.group(2)
        if weight is None or experiment is None:
            raise ValueError('Incorrect file name argument')
        with open(file) as f:
            data = json.load(f)
            return seq(data.items()).map(lambda kv: {
                'experiment': experiment,
                'weight': weight,
                'result': kv[0].replace('Answer.', ''),
                'score': kv[1]
            })

    rows = seq(glob(path.join(stats_dir, '*.json'))).flat_map(parse_file).to_pandas()
    return rows


@click.group()
def cli():
    pass


@cli.command()
@click.argument('stats_dir')
@click.argument('output')
def plot(stats_dir, output):
    import seaborn as sns
    rows = parse_data(stats_dir)
    g = sns.factorplot(y='result', x='score', row='experiment', col='weight',
                       data=rows, kind='bar', orient='h', ci=None, margin_titles=True,
                       order=ANSWER_PLOT_ORDER, size=10)
    g.savefig(output, format='png')


@cli.command()
@click.option('--min-count', default=5)
@click.option('--qdb', default='data/internal/non_naqt.db')
@click.argument('pred_file')
@click.argument('meta_file')
@click.argument('output')
def generate(min_count, qdb, pred_file, meta_file, output):
    database = QuestionDatabase(qdb)
    data = load_data(pred_file, meta_file, database)
    dan_answers = set(database.page_by_count(min_count=min_count, exclude_test=True))
    answers = compute_answers(data, dan_answers)
    stats = compute_statistics(answers).cache()
    stats.to_json(output, root_array=False)
    pp = pprint.PrettyPrinter()
    pp.pprint(stats)


if __name__ == '__main__':
    cli()
