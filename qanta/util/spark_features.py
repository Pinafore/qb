import os
from functools import reduce
from typing import List, Dict, Tuple

from pyspark import RDD, SparkContext
from pyspark.sql import SparkSession, Row, DataFrame
from pyspark.sql.types import StructField, StructType, StringType, IntegerType

from qanta import logging
from qanta.util.constants import VW_FOLDS, FEATURE_NAMES
from qanta.datasets.quiz_bowl import QuestionDatabase
from qanta.preprocess import format_guess

log = logging.get(__name__)

SCHEMA = StructType([
    StructField('fold', StringType(), False),
    StructField('qnum', IntegerType(), False),
    StructField('sentence', IntegerType(), False),
    StructField('token', IntegerType(), False),
    StructField('guess', StringType(), False),
    StructField('feature_name', StringType(), False),
    StructField('feature_value', StringType(), False)
])


def create_output(path: str):
    df = read_dfs(path).cache()
    question_db = QuestionDatabase()
    answers = question_db.all_answers()
    for qnum in answers:
        answers[qnum] = format_guess(answers[qnum])

    sc = SparkContext.getOrCreate()  # type: SparkContext
    b_answers = sc.broadcast(answers)

    def generate_string(group):
        rows = group[1]
        result = ""
        feature_values = []
        meta = None
        qnum = None
        guess = None
        for name in FEATURE_NAMES:
            named_feature_list = list(filter(lambda r: r.feature_name == name, rows))
            if len(named_feature_list) != 1:
                raise ValueError(
                    'Encountered more than one row when there should be exactly one row')
            named_feature = named_feature_list[0]
            if meta is None:
                qnum = named_feature.qnum
                guess = named_feature.guess
                meta = '{} {} {} {}'.format(
                    qnum,
                    named_feature.sentence,
                    named_feature.token,
                    guess
                )
            feature_values.append(named_feature.feature_value)
        assert '@' not in result, \
            '@ is a special character that is split on and not allowed in the feature line'

        vw_features = ' '.join(feature_values)
        if guess == b_answers.value[qnum]:
            vw_label = "1 '{} ".format(qnum)
        else:
            vw_label = "-1 '{} ".format(qnum)

        return vw_label + vw_features + '@' + meta

    for fold in VW_FOLDS:
        group_features(df.filter(df.fold == fold))\
            .map(generate_string)\
            .saveAsTextFile('output/vw_input/{0}.vw'.format(fold))


def read_dfs(path: str) -> DataFrame:
    sql_context = SparkSession.builder.getOrCreate()
    feature_dfs = {}  # type: Dict[Tuple[str, str], DataFrame]
    for fold in VW_FOLDS:
        for name in FEATURE_NAMES:
            log.info("Reading {fold} for {feature}".format(fold=fold, feature=name))
            filename = '{name}.parquet'.format(name=name)
            file_path = os.path.join(path, fold, filename)
            if not os.path.exists(file_path):
                log.info("File {file} does not exist".format(file=file_path))
                raise ValueError(
                    'Was not able to parse {file} since it does not exist'.format(file=file_path))
            else:
                feature_dfs[(fold, name)] = sql_context.read.load(file_path)

    return reduce(lambda x, y: x.union(y), feature_dfs.values())


def group_features(df: DataFrame) -> RDD:
    def seq_op(comb_value: List[Row], value: Row):
        if comb_value is None:
            return [value]
        else:
            comb_value.append(value)
            return comb_value

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

    grouped_rdd = df.rdd\
        .map(lambda r: ((r.qnum, r.sentence, r.token, r.guess), r))\
        .aggregateByKey(None, seq_op, comb_op)
    return grouped_rdd
