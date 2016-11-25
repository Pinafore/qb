from typing import List, NamedTuple
import os

from pyspark import SparkContext
from pyspark.sql import SparkSession, Row

import pandas as pd

from qanta import logging
from qanta.guesser.abstract import AbstractGuesser

from qanta.util.qdb import Question, QuestionDatabase
from qanta.util.io import safe_path
from qanta.util import constants as c
from qanta.util.spark_features import SCHEMA

from qanta.extractors.label import Labeler
from qanta.extractors.lm import LanguageModel
from qanta.extractors.deep import DeepExtractor
from qanta.extractors.classifier import Classifier
from qanta.extractors.wikilinks import WikiLinks
from qanta.extractors.mentions import Mentions
from qanta.extractors.answer_present import AnswerPresent
from qanta.extractors.text import TextExtractor


log = logging.get(__name__)
Task = NamedTuple('Task', [('question', Question), ('guess_df', pd.DataFrame)])


def instantiate_feature(feature_name: str):
    """
    @param feature_name: The feature to instantiate
    @param question_db: question database
    """

    log.info('Loading feature {} ...'.format(feature_name))
    if feature_name == "lm":
        feature = LanguageModel()
    elif feature_name == "deep":

        feature = DeepExtractor()
    elif feature_name == "wikilinks":
        feature = WikiLinks()
    elif feature_name == "answer_present":
        feature = AnswerPresent()
    elif feature_name == "label":
        feature = Labeler()
    elif feature_name == "classifier":
        feature = Classifier()
    elif feature_name == "mentions":
        feature = Mentions()
    elif feature_name == 'text':
        feature = TextExtractor()
    else:
        log.info("Don't know what to do with %s" % feature_name)
        raise ValueError("Wrong feature type")
    log.info("done")
    return feature


def spark_batch(sc: SparkContext, feature_names: List[str]):
    sql_context = SparkSession.builder.getOrCreate()

    log.info("Loading Questions")
    question_db = QuestionDatabase()
    question_map = question_db.all_questions()

    log.info("Loading Guesses")
    guess_df = None
    for guesser_class, _ in c.GUESSER_LIST:
        input_path = os.path.join(c.GUESSER_TARGET_PREFIX, guesser_class)
        if guess_df is None:
            guess_df = AbstractGuesser.load_guesses(input_path)
        else:
            new_guess_df = AbstractGuesser.load_guesses(input_path)
            guess_df = pd.concat([guess_df, new_guess_df])

    log.info("Number of guesses for feature extraction: {0}".format(len(guess_df)))

    tasks = []  # type: List[Task]
    for name, group in guess_df.groupby(['qnum', 'sentence', 'token']):
        qnum = int(name[0])
        question = question_map[qnum]
        tasks.append(Task(question, group))

    log.info('Number of tasks (unique qnum/sentence/token triplets): {}'.format(len(tasks)))

    log.info('Loading features: {}'.format(feature_names))
    features = {name: instantiate_feature(name) for name in feature_names}
    b_features = sc.broadcast(features)

    def f_eval(x: Task) -> List[Row]:
        return evaluate_feature_question(x, b_features)

    log.info("Beginning feature job")
    # Hand tuned value of 5000 to keep the task size below recommended 100KB
    feature_rdd = sc.parallelize(tasks, 5000 * len(feature_names)).flatMap(f_eval)

    feature_df = sql_context.createDataFrame(feature_rdd, SCHEMA).cache()

    log.info("Beginning write job")
    for fold in c.VW_FOLDS:
        feature_df_with_fold = feature_df.filter('fold = "{0}"'.format(fold)).cache()
        for name in feature_names:
            filename = safe_path('output/features/{}/{}.parquet'.format(fold, name))
            feature_df_with_fold\
                .filter('feature_name = "{}"'.format(name))\
                .write\
                .partitionBy('qnum')\
                .parquet(filename, mode='overwrite')
        feature_df_with_fold.unpersist()
    log.info("Computation Completed, stopping Spark")


def evaluate_feature_question(task: Task, b_features) -> List[Row]:
    features = b_features.value
    question = task.question
    guess_df = task.guess_df
    result = []
    if len(guess_df) > 0:
        for feature_name in features:
            feature_generator = features[feature_name]
            # guess_df is dataframe that contains values that are explicitly unique by
            # (qnum, sentence, token).
            #
            # This means that it is guaranteed that qnum, sentence, and token are all the same in
            # guess_df so it is safe and efficient to compute the text before iterating over guesses
            # as long as there is at least one guess. Additionally since a question is only ever
            # in one fold getting the fold is safe as well.
            first_row = guess_df.iloc[0]

            # Must cast numpy int64 to int for spark
            qnum = int(question.qnum)
            sentence = int(first_row.sentence)
            token = int(first_row.token)
            fold = first_row.fold
            text = question.get_text(sentence, token)
            feature_values = feature_generator.score_guesses(guess_df.guess, text)

            for f_value, guess in zip(feature_values, guess_df.guess):
                row = Row(
                    fold,
                    qnum,
                    sentence,
                    token,
                    guess,
                    feature_name,
                    f_value
                )
                result.append(row)
    return result
