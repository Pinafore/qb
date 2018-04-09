"""
CLI utilities for QANTA
"""

import json
import sqlite3
from os import path
import click
from sklearn.model_selection import train_test_split

from qanta import qlogging
from qanta.util.environment import ENVIRONMENT
from qanta.datasets.quiz_bowl import QuestionDatabase, Question, QB_QUESTION_DB
from qanta.guesser.abstract import AbstractGuesser
from qanta.util.io import safe_open, shell
from qanta.hyperparam import expand_config
from qanta.update_db import write_answer_map, merge_answer_mapping


log = qlogging.get(__name__)

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
def main():
    log.info("QANTA starting with configuration:")
    for k, v in ENVIRONMENT.items():
        log.info("{0}={1}".format(k, v))


@main.command()
@click.option('--fold', multiple=True, default=['guess_train', 'guessdev'])
@click.option('--merged_output', is_flag=True)
@click.argument('output_dir')
def export_db(fold, merged_output, output_dir):
    fold_set = set(fold)
    db = QuestionDatabase()
    if not db.location.endswith('non_naqt.db'):
        raise ValueError('Will not export naqt.db to json format to prevent data leaks')
    log.info(f'Outputing data for folds: {fold_set}')
    questions = [q for q in db.all_questions().values() if q.fold in fold_set]

    def to_example(question: Question):
        sentences = [question.text[i] for i in range(len(question.text))]
        return {
            'qnum': question.qnum,
            'sentences': sentences,
            'page': question.page,
            'fold': question.fold
        }

    if merged_output:
        log.info(f'Writing output to: {path.join(output_dir, "quiz-bowl.all.json")}')
        with safe_open(path.join(output_dir, 'quiz-bowl.all.json'), 'w') as f:
            json.dump({'questions': [to_example(q) for q in questions]}, f)
    else:
        all_train = [to_example(q) for q in questions if 'train' in q.fold]
        train, val = train_test_split(all_train, train_size=.9, test_size=.1)
        dev = [to_example(q) for q in questions if 'dev' in q.fold]

        log.info(f'Writing output to: {output_dir}/*')
        with safe_open(path.join(output_dir, 'quiz-bowl.train.json'), 'w') as f:
            json.dump({'questions': train}, f)

        with safe_open(path.join(output_dir, 'quiz-bowl.val.json'), 'w') as f:
            json.dump({'questions': val}, f)

        with safe_open(path.join(output_dir, 'quiz-bowl.dev.json'), 'w') as f:
            json.dump({'questions': dev}, f)


@main.command()
@click.option('--host', default='0.0.0.0')
@click.option('--port', default=5000)
@click.option('--debug', default=False)
@click.argument('guessers', nargs=-1)
def guesser_api(host, port, debug, guessers):
    if debug:
        log.warn('WARNING: debug mode in flask can expose environment variables, including AWS keys, NEVER use this when the API is exposed to the web')
        log.warn('Confirm that you would like to enable flask debugging')
        confirmation = input('yes/no:\n').strip()
        if confirmation != 'yes':
            raise ValueError('Most confirm enabling debug mode')

    AbstractGuesser.multi_guesser_web_api(guessers, host=host, port=port, debug=debug)


def run_guesser(n_times, workers, guesser_qualified_class):
    for _ in range(n_times):
        if 'qanta.guesser' not in guesser_qualified_class:
            log.error('qanta.guesser not found in guesser_qualified_class, this is likely an error, exiting.')
            return
        shell('rm -rf /tmp/qanta')
        shell(f'rm -rf output/guesser/{guesser_qualified_class}')
        shell(f'luigi --local-scheduler --module qanta.pipeline.guesser --workers {workers} AllSingleGuesserReports')


@main.command()
@click.option('--n_times', default=1)
@click.option('--workers', default=1)
@click.argument('guesser_qualified_class')
def guesser_pipeline(n_times, workers, guesser_qualified_class):
    run_guesser(n_times, workers, guesser_qualified_class)


@main.command()
@click.argument('output_dir')
def generate_additional_answer_mappings(output_dir):
    write_answer_map(output_dir)


@main.command()
@click.argument('source_db')
@click.argument('output_db')
@click.argument('answer_map_path')
@click.argument('page_assignments_path')
def db_merge_answers(source_db, output_db, answer_map_path, page_assignments_path):
    with open(answer_map_path) as f:
        answer_map = json.load(f)['answer_map']
    merge_answer_mapping(source_db, answer_map, output_db, page_assignments_path)


@main.command()
@click.option('--n', default=20)
def sample_answer_pages(n):
    """
    Take a random sample of n questions, then return their answers and pages
    formatted for latex in the journal paper
    """
    conn = sqlite3.connect(QB_QUESTION_DB)
    c = conn.cursor()
    rows = c.execute(f'select answer, page from questions order by random() limit {n}')
    latex_format = r'{answer} & {page}\\ \hline'
    for answer, page in rows:
        answer = answer.replace('{', r'\{').replace('}', r'\}').replace('_', r'\_')
        if page == '':
            page = r'\textbf{No Mapping Found}'
        else:
            page = page.replace('{', r'\{').replace('}', r'\}').replace('_', r'\_')
        print(latex_format.format(answer=answer, page=page))


@main.command()
@click.argument('base_file')
@click.argument('hyper_file')
@click.argument('output_file')
def hyper_to_conf(base_file, hyper_file, output_file):
    expand_config(base_file, hyper_file, output_file)


if __name__ == '__main__':
    main()
