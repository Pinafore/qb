import click
from pyspark import SparkConf, SparkContext

from qanta.util.constants import FEATURE_NAMES
from util.environment import QB_QUESTION_DB, QB_GUESS_DB, QB_SPARK_MASTER
from util import spark_features
import extract_features as ef

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
def spark():
    pass


def create_spark_context(app_name="Quiz Bowl", lm_memory=False, profile=False):
    spark_conf = SparkConf()
    if lm_memory:
        spark_conf = spark_conf.set('spark.max.cores', 12).set('spark.executor.cores', 12)
    if profile:
        spark_conf = spark_conf.set('spark.python.profile', True)
    return SparkContext(appName=app_name, master=QB_SPARK_MASTER, conf=spark_conf)


@spark.command()
@click.argument('features', nargs=-1, type=click.Choice(FEATURE_NAMES), required=True)
@click.option('--lm-memory', is_flag=True)
@click.option('--profile', is_flag=True)
def extract_features(**kwargs):
    sc = create_spark_context(
        app_name='Quiz Bowl: ' + ' '.join(kwargs['features']),
        lm_memory=kwargs['lm_memory'],
        profile=kwargs['profile']
    )
    ef.spark_execute(sc, kwargs['features'], QB_QUESTION_DB, QB_GUESS_DB)


@spark.command()
def merge_features(**kwargs):
    sc = create_spark_context(app_name='Quiz Bowl Merge')
    spark_features.create_output(sc, '/home/ubuntu/output/features')


if __name__ == '__main__':
    spark()
