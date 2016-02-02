import extract_features as ef
import click
from util.environment import QB_QUESTION_DB, QB_GUESS_DB, QB_SPARK_MASTER
from pyspark import SparkConf, SparkContext

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
def spark():
    pass


def create_spark_context():
    spark_conf = SparkConf().set('spark.max.cores', 12).set('spark.executor.cores', 12)
    return SparkContext(appName="QuizBowl", master=QB_SPARK_MASTER, conf=spark_conf)


@spark.command()
def extract_features(**kwargs):
    ef.spark_execute(create_spark_context(), QB_QUESTION_DB, QB_GUESS_DB)


@spark.command()
def merge_features(**kwargs):
    pass


if __name__ == '__main__':
    spark()
