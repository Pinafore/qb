import os

QB_ROOT = os.getenv('QB_ROOT')

QB_SPARK_MASTER = os.getenv('QB_SPARK_MASTER')


def data_path(other_path):
    if QB_ROOT:
        return os.path.join(QB_ROOT, other_path)
    else:
        return other_path

QB_WIKI_LOCATION = data_path('data/external/wikipedia')

if os.path.exists('data/internal/naqt.db'):
    QB_QUESTION_DB = data_path('data/internal/naqt.db')
else:
    QB_QUESTION_DB = data_path('data/internal/non_naqt.db')

ENVIRONMENT = dict(
    QB_ROOT=QB_ROOT,
    QB_QUESTION_DB=QB_QUESTION_DB,
    QB_SPARK_MASTER=QB_SPARK_MASTER
)