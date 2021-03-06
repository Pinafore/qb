import os
import multiprocessing
import boto3
from botocore.exceptions import NoCredentialsError
from functools import lru_cache

QB_ROOT = os.getenv("QB_ROOT")
QB_SPARK_MASTER = os.getenv("QB_SPARK_MASTER", "local[*]")
QB_MAX_CORES = os.getenv("QB_MAX_CORES", multiprocessing.cpu_count())


@lru_cache()
def is_aws_authenticated():
    try:
        boto3.client("iam").get_user()
        return True
    except:
        return False


def data_path(other_path):
    if QB_ROOT:
        return os.path.join(QB_ROOT, other_path)
    else:
        return other_path


NAQT_QBML_DIR = data_path("data/internal/naqt_qbml/")
BONUS_QUESTION_DB = data_path("data/internal/bonus.db")  # small database
BONUS_QUESTION_PKL = data_path("data/internal/bonus.pkl")  # large database from naqt
BONUS_ANSWER_PAGES = data_path("data/internal/bonus_answer_pages.pkl")
BONUS_PAIRS_JSON = data_path("data/internal/bonus_pairs.json")

ENVIRONMENT = dict(QB_ROOT=QB_ROOT, QB_SPARK_MASTER=QB_SPARK_MASTER)
