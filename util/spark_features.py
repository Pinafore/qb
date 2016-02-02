from pyspark.sql import SQLContext
from pyspark.sql.types import StructField, StructType, StringType, IntegerType

from util.constants import FOLDS
import os

SCHEMA = StructType([
    StructField('feature_name', StringType(), True),
    StructField('fold', StringType(), True),
    StructField('page', StringType(), True),
    StructField('qnum', IntegerType(), True),
    StructField('meta', StringType(), True),
    StructField('feat', StringType(), True)
])


def merge_features(sc, path, granularity='sentence'):
    sql_context = SQLContext(sc)
    feature_names = ['label', 'ir', 'lm', 'deep', 'answer_present', 'text', 'classifier',
                     'wikilinks']
    df = None
    for fold in FOLDS:
        for name in feature_names:
            filename = '{granularity}.{name}.parquet'.format(granularity=granularity, name=name)
            new_df = sql_context.read.load(os.path.join(path, fold, filename))
            if df:
                df.unionAll(new_df)
            else:
                df = new_df
    df
