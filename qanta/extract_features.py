from collections import namedtuple
import os
from unidecode import unidecode
from random import shuffle

from pyspark import SparkContext
from pyspark.sql import SparkSession, Row

from qanta import logging
from qanta.util.build_whoosh import text_iterator
from qanta.util.guess import GuessList
from qanta.util import constants as C
from qanta.util.constants import FOLDS, MIN_APPEARANCES
from qanta.util.qdb import QuestionDatabase
from qanta.util.spark_features import SCHEMA

from qanta.datasets.quiz_bowl import QuizBowlDataset

from qanta.extractors.label import Labeler
from qanta.extractors.lm import LanguageModel
from qanta.extractors.deep import DeepExtractor
from qanta.extractors.classifier import Classifier
from qanta.extractors.wikilinks import WikiLinks
from qanta.extractors.mentions import Mentions
from qanta.extractors.answer_present import AnswerPresent
from qanta.extractors.text import TextExtractor


log = logging.get(__name__)
Task = namedtuple('Task', ['question', 'guesses'])


def feature_lines(question, guesses_needed, granularity, feature_generator):
    for sentence, token in guesses_needed:
        if granularity == "sentence" and token > 0:
            continue

        # Set metadata so the labeler can create ids and weights
        guess_size = len(guesses_needed[(sentence, token)])
        feature_generator.set_metadata(question.page, question.category, question.qnum, sentence,
                                       token, guess_size, question)

        for feat, guess in feature_generator.score_guesses(
                guesses_needed[(sentence, token)],
                question.get_text(sentence, token)):
            yield sentence, token, guess, feat


def instantiate_feature(feature_name: str, question_db: QuestionDatabase):
    """
    @param feature_name: The feature to instantiate
    @param question_db: question database
    """

    feature = None
    print("Loading feature %s ..." % feature_name)
    if feature_name == "lm":
        feature = LanguageModel()
    elif feature_name == "deep":
        page_dict = {}
        for page in question_db.get_all_pages():
            page_dict[page.lower().replace(' ', '_')] = page
        feature = DeepExtractor(
            C.DEEP_DAN_CLASSIFIER_TARGET,
            C.DEEP_DAN_PARAMS_TARGET,
            C.DEEP_VOCAB_TARGET,
            C.NERS_PATH,
            page_dict
        )
    elif feature_name == "wikilinks":
        feature = WikiLinks()
    elif feature_name == "answer_present":
        feature = AnswerPresent()
    elif feature_name == "label":
        feature = Labeler(question_db)
    elif feature_name == "classifier":
        feature = Classifier()
    elif feature_name == "mentions":
        answers = set(x for x, y in text_iterator(
            False, "", False, question_db, False, "", limit=-1, min_pages=MIN_APPEARANCES))
        feature = Mentions(answers)
    elif feature_name == 'text':
        feature = TextExtractor()
    else:
        log.info("Don't know what to do with %s" % feature_name)
        raise ValueError("Wrong feature type")
    log.info("done")
    return feature


def spark_batch(sc: SparkContext, feature_names, question_db_path: str, guess_db: str,
                granularity='sentence'):
    sql_context = SparkSession.builder.getOrCreate()
    question_db = QuestionDatabase(question_db_path)

    log.info("Loading Questions")
    qb_dataset = QuizBowlDataset(5, question_db_path)
    questions = qb_dataset.questions_in_folds(['dev', 'devtest', 'test'])

    log.info("Loading Guesses")
    guess_list = GuessList(guess_db)
    guess_lookup = guess_list.all_guesses(allow_train=True)

    log.info("Loading tasks")
    tasks = [Task(q, guess_lookup[q.qnum]) for q in questions]
    shuffle(tasks)
    log.info("Number of tasks: {0}".format(len(tasks)))

    features = {name: instantiate_feature(name, question_db) for name in feature_names}

    b_features = sc.broadcast(features)

    def f_eval(x):
        return evaluate_feature_question(x, b_features, granularity)

    log.info("Beginning feature job")
    feature_rdd = sc.parallelize(tasks)\
        .repartition(150 * len(feature_names))\
        .flatMap(f_eval)

    feature_df = sql_context.createDataFrame(feature_rdd, SCHEMA).cache()
    feature_df.count()
    log.info("Beginning write job")
    for fold in FOLDS:
        feature_df_with_fold = feature_df.filter('fold = "{0}"'.format(fold)).cache()
        for name in feature_names:
            filename = 'output/features/{0}/sentence.{1}.parquet'.format(fold, name)
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            feature_df_with_fold.filter('feature_name = "{0}"'.format(name))\
                .write.save(filename, mode='overwrite')
        feature_df_with_fold.unpersist()
    log.info("Computation Completed, stopping Spark")


def evaluate_feature_question(task, b_features, granularity):
    features = b_features.value
    question = task.question
    result = []
    for feature_name in features:
        feature_generator = features[feature_name]
        for sentence, token, guess, feat in feature_lines(
                question, task.guesses, granularity, feature_generator):
            result.append(
                Row(
                    feature_name,
                    question.fold,
                    guess,
                    question.qnum,
                    sentence,
                    token,
                    '%i\t%i\t%i\t%s' % (question.qnum, sentence, token, unidecode(guess)),
                    feat
                )
            )
    return result
