import os
import multiprocessing
import boto3
from botocore.exceptions import NoCredentialsError
from functools import lru_cache

QB_ROOT = os.getenv('QB_ROOT')

QB_SPARK_MASTER = os.getenv('QB_SPARK_MASTER', 'local[*]')

QB_MAX_CORES = os.getenv('QB_MAX_CORES', multiprocessing.cpu_count())

QB_TB_HOSTNAME = os.getenv('QB_TB_HOSTNAME', 'localhost')
QB_TB_PORT = int(os.getenv('QB_TB_PORT', 6007))

TAGME_GCUBE_TOKEN = os.getenv('TAGME_GCUBE_TOKEN')



@lru_cache()
def is_aws_authenticated():
    try:
        boto3.client('iam').get_user()
        return True
    except NoCredentialsError:
        return False




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
