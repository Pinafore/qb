import os

QB_ROOT = os.getenv('QB_ROOT')

QB_QUESTION_DB = os.getenv('QB_QUESTION_DB', 'data/internal/questions.db')
QB_GUESS_DB = os.getenv('QB_GUESS_DB', 'output/guesses.db')

QB_SPARK_MASTER = os.getenv('QB_SPARK_MASTER')
QB_STREAMING_CORES = os.getenv('QB_STREAMING_CORES', 12)

QB_WIKI_LOCATION = os.getenv('QB_WIKI_LOCATION', 'data/external/wikipedia')

QB_API_DOMAIN = os.getenv('QB_API_DOMAIN', '')
QB_API_USER_ID = int(os.getenv('QB_API_USER_ID', 1))
QB_API_KEY = os.getenv('QB_API_KEY', '')


def data_path(other_path):
    if QB_ROOT:
        return os.path.join(QB_ROOT, other_path)
    else:
        return other_path
