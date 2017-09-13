import os
import multiprocessing

QB_ROOT = os.getenv('QB_ROOT')

QB_SPARK_MASTER = os.getenv('QB_SPARK_MASTER')

QB_MAX_CORES = os.getenv('QB_MAX_CORES', multiprocessing.cpu_count())

TAGME_GCUBE_TOKEN = os.getenv('TAGME_GCUBE_TOKEN')


def data_path(other_path):
    if QB_ROOT:
        return os.path.join(QB_ROOT, other_path)
    else:
        return other_path

if os.path.exists('data/internal/naqt.db'):
    QB_QUESTION_DB = data_path('data/internal/naqt.db')
else:
    QB_QUESTION_DB = data_path('data/internal/non_naqt.db')

ENVIRONMENT = dict(
    QB_ROOT=QB_ROOT,
    QB_QUESTION_DB=QB_QUESTION_DB,
    QB_SPARK_MASTER=QB_SPARK_MASTER,
    TAGME_GCUBE_TOKEN=TAGME_GCUBE_TOKEN
)
