import os
from functools import reduce
from typing import List, Dict, Tuple
from pyspark import SparkContext, RDD
from pyspark.sql import SQLContext, Row, DataFrame
from pyspark.sql.functions import udf
from pyspark.sql.types import StructField, StructType, StringType, IntegerType

from util.constants import FOLDS, FEATURE_NAMES

NEGATIVE_WEIGHTS = [2., 4., 8., 16., 32., 64.]


SCHEMA = StructType([
    StructField('feature_name', StringType(), True),
    StructField('fold', StringType(), True),
    StructField('guess', StringType(), True),
    StructField('qnum', IntegerType(), True),
    StructField('sentence', IntegerType(), True),
    StructField('token', IntegerType(), True),
    StructField('meta', StringType(), True),
    StructField('feat', StringType(), True)
])


def create_output(sc: SparkContext, path: str, granularity='sentence'):
    df = read_dfs(sc, path, granularity=granularity).cache()
    for fold in FOLDS:
        filtered_df = df.filter('fold = "{0}"'.format(fold))
        grouped_rdd = group_features(filtered_df).cache()
        for weight in NEGATIVE_WEIGHTS:
            def generate_string(group):
                rows = group[1]
                result = ""
                for name in FEATURE_NAMES:
                    if name == 'label':
                        name = 'label{0}'.format(weight)
                    named_feature_list = list(filter(lambda r: r.feature_name == name, rows))
                    if len(named_feature_list) != 1:
                        continue
                    named_feature = named_feature_list[0]
                    result = result + '\t' + named_feature.feat
                return result

            output_rdd = grouped_rdd.map(generate_string)
            output_rdd.saveAsTextFile(
                '/home/ubuntu/output/vw_input/{0}/sentence.{1}.vw_input'.format(fold, int(weight)),
                compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")
        grouped_rdd.unpersist()


def read_dfs(sc: SparkContext, path: str, granularity='sentence') -> DataFrame:
    sql_context = SQLContext(sc)
    feature_dfs = {}  # type: Dict[Tuple[str, str], DataFrame]
    for fold in FOLDS:
        for name in FEATURE_NAMES:
            print("Reading {fold} for {feature}".format(fold=fold, feature=name))
            filename = '{granularity}.{name}.parquet'.format(granularity=granularity, name=name)
            file_path = os.path.join(path, fold, filename)
            if not os.path.exists(file_path):
                print("File {file} does not exist".format(file=file_path))
            else:
                feature_dfs[(fold, name)] = sql_context.read.load(file_path).cache()
                count = feature_dfs[(fold, name)].count()
                print("{count} rows read for {fold} and {name}".format(
                    count=count, fold=fold, name=name))

    reweight_df(feature_dfs)

    return reduce(lambda x, y: x.unionAll(y), feature_dfs.values())


def reweight_df(feature_dfs: Dict[Tuple[str, str], DataFrame]):
    for fold in FOLDS:
        for weight in NEGATIVE_WEIGHTS:
            def reweight(feat: str):
                feat_split = feat.split()
                label, neg_count_str = feat_split[0], feat_split[1]
                if int(label) == 1:
                    return feat
                else:
                    neg_count = int(neg_count_str)
                    return feat.replace(" %s '" % neg_count_str,
                                        " %f '" % (weight / float(neg_count)))
            weight_udf = udf(reweight, StringType())
            label_str = 'label{0}'.format(weight)
            name_udf = udf(lambda r: label_str)
            temp_df = feature_dfs[(fold, 'label')]
            key = (fold, label_str)
            feature_dfs[key] = temp_df\
                .withColumn('feature_name', name_udf(temp_df.feature_name))\
                .withColumn('feat', weight_udf(temp_df.feat)).cache()
            feature_dfs[key].count()
        del feature_dfs[(fold, 'label')]


def group_features(df: DataFrame) -> RDD:
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

    grouped_rdd = df.rdd.map(lambda r: ((r.guess, r.qnum, r.sentence, r.token), r))\
        .aggregateByKey(None, seq_op, comb_op)
    return grouped_rdd
