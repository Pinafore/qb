from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from qanta.util.environment import QB_SPARK_MASTER


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
