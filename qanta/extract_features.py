from collections import defaultdict, namedtuple
from unidecode import unidecode
from random import shuffle
from multiprocessing import Pool, cpu_count

from pyspark import SparkContext
from pyspark.sql import SQLContext, Row

from qanta import logging
from qanta.util.build_whoosh import text_iterator
from qanta.util.guess import GuessList
from qanta.util.constants import FOLDS, MIN_APPEARANCES, CLM_PATH
from qanta.util.qdb import QuestionDatabase
from qanta.util.environment import data_path, QB_QUESTION_DB, QB_GUESS_DB
from qanta.util.spark_features import SCHEMA

from qanta.extractors.label import Labeler
from qanta.extractors.lm import LanguageModel
from qanta.extractors.deep import DeepExtractor
from qanta.extractors.classifier import Classifier
from qanta.extractors.wikilinks import WikiLinks
from qanta.extractors.mentions import Mentions
from qanta.extractors.answer_present import AnswerPresent


log = logging.get(__name__)
Task = namedtuple('Task', ['question', 'guesses'])


def feature_lines(question, guesses_needed, granularity, feature_generator):
    # Guess we might have already
    # It has the structure:
    # guesses[(sent, token)][page][feat] = value

    for sentence, token in guesses_needed:
        if granularity == "sentence" and token > 0:
            continue

        # Set metadata so the labeler can create ids and weights
        guess_size = len(guesses_needed[(sentence, token)])
        feature_generator.set_metadata(question.page, question.category, question.qnum, sentence,
                                       token, guess_size, question)

        for guess in guesses_needed[(sentence, token)]:
            feat = feature_generator.vw_from_title(guess, question.get_text(sentence, token))
            yield sentence, token, guess, feat


def instantiate_feature(feature_name, questions, deep_data="output/deep"):
    """
    @param feature_name: The feature to instantiate
    @param questions: question database
    """

    feature = None
    print("Loading feature %s ..." % feature_name)
    if feature_name == "lm":
        feature = LanguageModel(data_path(CLM_PATH))
    elif feature_name == "deep":
        print("from %s" % deep_data)
        page_dict = {}
        for page in questions.get_all_pages():
            page_dict[page.lower().replace(' ', '_')] = page
        feature = DeepExtractor(
            "%s/classifier" % deep_data,
            "%s/params" % deep_data,
            "%s/vocab" % deep_data,
            "data/internal/common/ners",
            page_dict
        )
    elif feature_name == "wikilinks":
        feature = WikiLinks()
    elif feature_name == "answer_present":
        feature = AnswerPresent()
    elif feature_name == "label":
        feature = Labeler(questions)
    elif feature_name == "classifier":
        feature = Classifier(data_path('data/classifier/bigrams.pkl'), questions)
    elif feature_name == "mentions":
        answers = set(x for x, y in text_iterator(
            False, "", False, questions, False, "", limit=-1, min_pages=MIN_APPEARANCES))
        feature = Mentions(answers)
    else:
        print("Don't know what to do with %s" % feature_name)
    print("done")
    return feature


def create_guesses_for_question(question, deep_feature, word_skip=-1):
    final_guesses = defaultdict(dict)

    # Gather all the guesses
    for sentence, token, text in question.partials(word_skip):
        # We have problems at the very start
        if sentence == 0 and token == word_skip:
            continue

        guesses = deep_feature.text_guess(text)
        for guess in guesses:
            final_guesses[(sentence, token)][guess] = guesses[guess]
        # add the correct answer if this is a training document and
        if question.fold == "train" and question.page not in guesses:
            final_guesses[(sentence, token)][question.page] = deep_feature.score_one_guess(
                question.page, text)

    return final_guesses


def spark_batch(sc: SparkContext, feature_names, question_db, guess_db, granularity='sentence'):
    sql_context = SQLContext(sc)
    question_db = QuestionDatabase(question_db)

    print("Loading Questions")
    questions = question_db.guess_questions()

    print("Loading Guesses")
    guess_list = GuessList(guess_db)
    guess_lookup = guess_list.all_guesses(allow_train=True)

    print("Loading tasks")
    tasks = [Task(q, guess_lookup[q.qnum]) for q in questions]
    shuffle(tasks)
    print("Number of tasks: {0}".format(len(tasks)))

    features = {name: instantiate_feature(name, question_db) for name in feature_names}

    b_features = sc.broadcast(features)

    def f_eval(x):
        return evaluate_feature_question(x, b_features, granularity)

    print("Beginning feature job")
    feature_rdd = sc.parallelize(tasks)\
        .repartition(150 * len(feature_names))\
        .flatMap(f_eval)

    feature_df = sql_context.createDataFrame(feature_rdd, SCHEMA).repartition(30).cache()
    feature_df.count()
    print("Beginning write job")
    for fold in FOLDS:
        feature_df_with_fold = feature_df.filter('fold = "{0}"'.format(fold)).cache()
        for name in feature_names:
            filename = '/home/ubuntu/qb/data/features/{0}/sentence.{1}.parquet'.format(fold, name)
            feature_df_with_fold.filter('feature_name = "{0}"'.format(name))\
                .write.save(filename, mode='overwrite')
        feature_df_with_fold.unpersist()
    print("Computation Completed, stopping Spark")
    sc.stop()


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


def parallel_generate_guesses(task):
    return task[0].qnum, task[0].fold, create_guesses_for_question(task[0], task[1])


def create_guesses(guess_db_path, processes=cpu_count()):
    q_db = QuestionDatabase(QB_QUESTION_DB)
    guess_list = GuessList(guess_db_path)

    deep_feature = instantiate_feature('deep', q_db)
    questions = q_db.guess_questions()
    tasks = []
    for q in questions:
        tasks.append((q, deep_feature))

    with Pool(processes=processes) as pool:
        question_guesses = pool.imap(parallel_generate_guesses, tasks)
        i, n = 0, len(tasks)
        log.info("Guess generation starting for {0} questions".format(n))
        for qnum, fold, guesses in question_guesses:
            guess_list.save_guesses('deep', qnum, fold, guesses)
            log.info("Progress: {0} / {1} questions completed".format(i, n))
            i += 1

    log.info("Guess generation completed, generating indices")
    guess_list.create_indexes()
    log.info("Guess generation done")
