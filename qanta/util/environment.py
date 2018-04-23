import os
import multiprocessing
import boto3
from botocore.exceptions import NoCredentialsError
from functools import lru_cache

QB_ROOT = os.getenv('QB_ROOT')
QB_SPARK_MASTER = os.getenv('QB_SPARK_MASTER', 'local[*]')
QB_MAX_CORES = os.getenv('QB_MAX_CORES', multiprocessing.cpu_count())


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


ENVIRONMENT = dict(
    QB_ROOT=QB_ROOT,
    QB_SPARK_MASTER=QB_SPARK_MASTER
)
