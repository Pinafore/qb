import extract_features
from util.environment import QB_QUESTION_DB, QB_GUESS_DB, QB_SPARK_MASTER

if __name__ == '__main__':
    extract_features.spark_execute(QB_SPARK_MASTER, QB_QUESTION_DB, QB_GUESS_DB)
