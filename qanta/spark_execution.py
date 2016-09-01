import click
from pyspark import SparkConf, SparkContext

from qanta.util.constants import FEATURE_NAMES
from qanta.util.environment import QB_QUESTION_DB, QB_GUESS_DB, QB_SPARK_MASTER
from qanta.util import spark_features
import qanta.extract_features as ef

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
def cli():
    pass


def create_spark_context(app_name="Quiz Bowl", lm_memory=False, profile=False):
    spark_conf = SparkConf()
    if lm_memory:
        pass
        # spark_conf = spark_conf.set('spark.max.cores', 30).set('spark.executor.cores', 30)
    if profile:
        spark_conf = spark_conf.set('spark.python.profile', True)
    spark_conf = spark_conf.set('spark.akka.frameSize', 300)
    return SparkContext(appName=app_name, master=QB_SPARK_MASTER, conf=spark_conf)


def extract_features(features, lm_memory=False, profile=False):
    sc = create_spark_context(
        app_name='Quiz Bowl: ' + ' '.join(features),
        lm_memory=lm_memory,
        profile=profile
    )
    ef.spark_batch(sc, features, QB_QUESTION_DB, QB_GUESS_DB)


@cli.command(name='extract_features')
@click.argument('features', nargs=-1, type=click.Choice(FEATURE_NAMES), required=True)
@click.option('--lm-memory', is_flag=True)
@click.option('--profile', is_flag=True)
def extract_features_cli(**kwargs):
    extract_features(kwargs['features'], lm_memory=kwargs['lm_memory'], profile=kwargs['profile'])


def merge_features():
    sc = create_spark_context(app_name='Quiz Bowl Merge')
    spark_features.create_output(sc, 'output/features')


@cli.command(name='merge_features')
def merge_features_cli(**kwargs):
    merge_features()


if __name__ == '__main__':
    cli()
