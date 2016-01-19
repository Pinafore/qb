import os
import extract_features
from extractors.lm import LanguageModel, JelinekMercerLanguageModel, DistCounter

if __name__ == '__main__':
    question_db = os.environ.get('QB_QUESTION_DB')
    guess_db = os.environ.get('QB_GUESS_DB')
    spark_master = os.environ.get('QB_SPARK_MASTER')
    extract_features.spark_execute(spark_master, question_db, guess_db)
