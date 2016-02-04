import os
from functools import reduce
from typing import List
from pyspark import SparkContext
from pyspark.sql import SQLContext, Row
from pyspark.sql.types import StructField, StructType, StringType, IntegerType

from util.constants import FOLDS, FEATURE_NAMES


SCHEMA = StructType([
    StructField('feature_name', StringType(), True),
    StructField('fold', StringType(), True),
    StructField('page', StringType(), True),
    StructField('qnum', IntegerType(), True),
    StructField('sentence', IntegerType(), True),
    StructField('token', IntegerType(), True),
    StructField('meta', StringType(), True),
    StructField('feat', StringType(), True)
])


def merge_features(sc: SparkContext, path: str, granularity='sentence'):
    sql_context = SQLContext(sc)
    feature_dfs = {}
    for fold in FOLDS:
        for name in FEATURE_NAMES:
            print("Reading {fold} for {feature}".format(fold=fold, feature=name))
            filename = '{granularity}.{name}.parquet'.format(granularity=granularity, name=name)
            file_path = os.path.join(path, fold, filename)
            if not os.path.exists(file_path):
                print("File {file} does not exist".format(file=file_path))
                continue
            new_rdd = sql_context.read.load(file_path).map(
                lambda r: Row(feature_name=name,
                              fold=fold,
                              page=r.page,
                              qnum=r.qnum,
                              meta=r.meta,
                              feat=r.feat))
            new_rdd.cache()
            if new_rdd.isEmpty():
                print("No rows found for {fold} and {feature}".format(fold=fold, feature=name))
                continue
            feature_dfs[(fold, name)] = sql_context.createDataFrame(new_rdd).cache()
            count = feature_dfs[(fold, name)].count()
            print("{count} rows read for {fold} and {name}".format(
                count=count, fold=fold, name=name))
    df = reduce(lambda x, y: x.unionAll(y), feature_dfs.values())

    def seq_op(comb_value: List[Row], value: Row):
        if comb_value is None:
            return [value]
        else:
            return [value] + comb_value

    def comb_op(left: List[Row], right: List[Row]):
        if left is None and right is None:
            return list()
        elif left is None:
            return right
        elif right is None:
            return left
        else:
            left.extend(right)
            return left

    grouped_rdd = df.rdd.map(lambda r: (r.page + '|' + str(r.qnum), r))\
        .aggregateByKey(None, seq_op, comb_op)
    return df, grouped_rdd
