import json
from os import path
import click
from sklearn.model_selection import train_test_split

from qanta import qlogging
from qanta.util.environment import ENVIRONMENT
from qanta.wikipedia.cached_wikipedia import web_initialize_file_cache
from qanta.datasets.quiz_bowl import QuestionDatabase, Question
from qanta.util.io import safe_open


log = qlogging.get(__name__)

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
def main():
    log.info("QANTA starting with configuration:")
    for k, v in ENVIRONMENT.items():
        log.info("{0}={1}".format(k, v))


@main.command()
@click.argument('wiki_cache')
def init_wiki_cache(wiki_cache):
    web_initialize_file_cache(wiki_cache)

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


if __name__ == '__main__':
    main()
