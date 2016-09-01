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


def create_spark_context(app_name="Quiz Bowl"):
    spark_conf = SparkConf()\
        .set('spark.akka.frameSize', 300)\
        .setAppName(app_name)\
        .setMaster(QB_SPARK_MASTER)
    return SparkContext.getOrCreate(spark_conf)


def extract_features(features):
    sc = create_spark_context(
        app_name='Quiz Bowl: ' + ' '.join(features)
    )
    ef.spark_batch(sc, features, QB_QUESTION_DB, QB_GUESS_DB)


@cli.command(name='extract_features')
@click.argument('features', nargs=-1, type=click.Choice(FEATURE_NAMES), required=True)
def extract_features_cli(**kwargs):
    extract_features(kwargs['features'])


def merge_features():
    sc = create_spark_context(app_name='Quiz Bowl Merge')
    spark_features.create_output('output/features')


@cli.command(name='merge_features')
def merge_features_cli(**kwargs):
    merge_features()


if __name__ == '__main__':
    cli()
