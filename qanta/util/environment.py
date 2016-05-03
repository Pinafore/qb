import os

QB_QUESTION_DB = os.getenv('QB_QUESTION_DB', 'data/questions.db')
QB_GUESS_DB = os.getenv('QB_GUESS_DB', 'data/guesses.db')
QB_SPARK_MASTER = os.getenv('QB_SPARK_MASTER')
QB_ROOT = os.getenv('QB_ROOT')


def data_path(other_path):
    if QB_ROOT:
        return os.path.join(QB_ROOT, other_path)
    else:
        return other_path
