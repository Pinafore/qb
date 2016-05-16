import os

QB_QUESTION_DB = os.getenv('QB_QUESTION_DB', 'data/questions.db')
QB_GUESS_DB = os.getenv('QB_GUESS_DB', 'data/guesses.db')
QB_SPARK_MASTER = os.getenv('QB_SPARK_MASTER')
QB_ROOT = os.getenv('QB_ROOT')
QB_STREAMING_CORES = os.getenv('QB_STREAMING_CORES', 12)

ENVIRONMENT = {
    'QB_QUESTION_DB': QB_QUESTION_DB,
    'QB_GUESS_DB': QB_GUESS_DB,
    'QB_SPARK_MASTER': QB_SPARK_MASTER,
    'QB_ROOT': QB_ROOT,
    'QB_STREAMING_CORES': QB_STREAMING_CORES
}


def data_path(other_path):
    if QB_ROOT:
        return os.path.join(QB_ROOT, other_path)
    else:
        return other_path
