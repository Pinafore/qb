from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

from qanta.util.environment import QB_SPARK_MASTER, QB_MAX_CORES
from qanta import logging


log = logging.get(__name__)


def create_spark_context(app_name="Quiz Bowl", configs=None) -> SparkContext:
    if QB_SPARK_MASTER != "":
        log.info("Spark master is %s" % QB_SPARK_MASTER)
        spark_conf = SparkConf()\
            .set('spark.rpc.message.maxSize', 300)\
            .setAppName(app_name)\
            .setMaster(QB_SPARK_MASTER)
    else:
        spark_conf = SparkConf()\
            .set('spark.rpc.message.maxSize', 300)\
            .setAppName(app_name)
    if configs is not None:
        for key, value in configs:
            if key in ('spark.executor.cores', 'spark.max.cores'):
                if value > QB_MAX_CORES:
                    log.info('Requested {r_cores} cores when the machine only has {n_cores} cores, reducing number of '
                             'cores to {n_cores}'.format(r_cores=value, n_cores=QB_MAX_CORES))
                    value = QB_MAX_CORES
            spark_conf = spark_conf.set(key, value)
    return SparkContext.getOrCreate(spark_conf)


def create_spark_session(app_name='Quiz Bowl', configs=None) -> SparkSession:
    create_spark_context(app_name=app_name, configs=configs)
    return SparkSession.builder.getOrCreate()
