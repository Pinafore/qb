import json
from os import path
import click
from sklearn.model_selection import train_test_split

from qanta import qlogging
from qanta.util.environment import ENVIRONMENT
from qanta.datasets.quiz_bowl import QuestionDatabase, Question
from qanta.guesser.abstract import AbstractGuesser
from qanta.util.io import safe_open, shell


log = qlogging.get(__name__)

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
def main():
    log.info("QANTA starting with configuration:")
    for k, v in ENVIRONMENT.items():
        log.info("{0}={1}".format(k, v))


@main.command()
@click.argument('output_dir')
def export_db(output_dir):
    db = QuestionDatabase()
    if not db.location.endswith('non_naqt.db'):
        raise ValueError('Will not export naqt.db to json format to prevent data leaks')
    questions = [q for q in db.all_questions().values() if q.fold in {'guesstrain', 'guessdev'}]

    def to_example(question: Question):
        sentences = [question.text[i] for i in range(len(question.text))]
        return {
            'qnum': question.qnum,
            'sentences': sentences,
            'page': question.page,
            'fold': question.fold
        }

    all_train = [to_example(q) for q in questions if q.fold == 'guesstrain']
    train, val = train_test_split(all_train, train_size=.9, test_size=.1)
    dev = [to_example(q) for q in questions if q.fold == 'guessdev']

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


@main.command()
@click.option('--n_times', default=1)
@click.option('--workers', default=1)
@click.argument('guesser_qualified_class')
def guesser_pipeline(n_times, workers, guesser_qualified_class):
    for _ in range(n_times):
        if 'qanta.guesser' not in guesser_qualified_class:
            log.error('qanta.guesser not found in guesser_qualified_class, this is likely an error, exiting.')
            return
        shell('rm -rf /tmp/qanta')
        shell(f'rm -rf output/guesser/{guesser_qualified_class}')
        shell(f'luigi --local-scheduler --module qanta.pipeline.guesser --workers {workers} AllSingleGuesserReports')


if __name__ == '__main__':
    main()
