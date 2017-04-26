import click
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

from qanta.util.constants import FEATURE_NAMES
from qanta.util.environment import QB_SPARK_MASTER
from qanta.util import spark_features
import qanta.extract_features as ef

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
def cli():
    pass


def create_spark_context(app_name="Quiz Bowl", configs=None) -> SparkContext:
    spark_conf = SparkConf()\
        .set('spark.rpc.message.maxSize', 300)\
        .setAppName(app_name)\
        .setMaster(QB_SPARK_MASTER)
    if configs is not None:
        for key, value in configs:
            spark_conf = spark_conf.set(key, value)
    return SparkContext.getOrCreate(spark_conf)


def create_spark_session(app_name='Quiz Bowl', configs=None) -> SparkSession:
    create_spark_context(app_name=app_name, configs=configs)
    return SparkSession.builder.getOrCreate()


def extract_features(features):
    if 'lm' in features:
        # This deals with out of memory problems when using the language model
        configs = [('spark.executor.cores', 10)]
    else:
        configs = None
    create_spark_context(
        app_name='Quiz Bowl: ' + ' '.join(features), configs=configs
    )
    ef.spark_batch(features)


def extract_guess_features():
    create_spark_context(
        app_name='Quiz Bowl: guessers',
        configs=[('spark.executor.cores', 10)]
    )
    ef.generate_guesser_feature()


@cli.command(name='extract_features')
@click.argument('features', nargs=-1, type=click.Choice(FEATURE_NAMES), required=True)
def extract_features_cli(**kwargs):
    extract_features(kwargs['features'])


def merge_features():
    create_spark_context(app_name='Quiz Bowl Merge')
    spark_features.create_output('output/features')


@cli.command(name='merge_features')
def merge_features_cli(**kwargs):
    merge_features()


if __name__ == '__main__':
    cli()
