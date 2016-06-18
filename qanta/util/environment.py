import os

QB_QUESTION_DB = os.getenv('QB_QUESTION_DB', 'data/questions.db')
QB_GUESS_DB = os.getenv('QB_GUESS_DB', 'output/guesses.db')
QB_SPARK_MASTER = os.getenv('QB_SPARK_MASTER')
QB_ROOT = os.getenv('QB_ROOT')
QB_STREAMING_CORES = os.getenv('QB_STREAMING_CORES', 12)
QB_WIKI_LOCATION = os.getenv('QB_WIKI_LOCATION', 'data/external/wikipedia')
QB_SOURCE_LOCATION = os.getenv('QB_SOURCE_LOCATION', 'data/internal/source')
QB_API_DOMAIN = os.getenv('QB_API_DOMAIN', '')
QB_API_USER_ID = int(os.getenv('QB_API_USER_ID', 1))
QB_API_KEY = os.getenv('QB_API_KEY', '')

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
