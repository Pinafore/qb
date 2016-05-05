from collections import defaultdict, namedtuple
from unidecode import unidecode
from random import shuffle
from typing import Dict
from multiprocessing import Pool, cpu_count

from pyspark import SparkContext
from pyspark.sql import SQLContext, Row
from pyspark.streaming import StreamingContext

from util.build_whoosh import text_iterator
from qanta.util.guess import GuessList
from qanta.util.constants import FOLDS, MIN_APPEARANCES, FEATURE_NAMES
from qanta.util.qdb import QuestionDatabase
from qanta.util.environment import data_path, QB_QUESTION_DB, QB_GUESS_DB
from qanta.util.spark_features import SCHEMA

from qanta.extractors.abstract import FeatureExtractor
from qanta.extractors.label import Labeler
from qanta.extractors.ir import IrExtractor
from qanta.extractors.lm import LanguageModel
from qanta.extractors.deep import DeepExtractor
from qanta.extractors.classifier import Classifier
from qanta.extractors.wikilinks import WikiLinks
from qanta.extractors.mentions import Mentions
from qanta.extractors.answer_present import AnswerPresent


HAS_GUESSES = set([e.has_guess() for e in [IrExtractor, LanguageModel, DeepExtractor,
                                           Classifier, AnswerPresent]])

Task = namedtuple('Task', ['question', 'guesses'])


def feature_lines(question, guesses_cached, guesses_needed, granularity, feature_generator):
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
            if guesses_cached is not None:
                feat = feature_generator.vw_from_score(guesses_cached[guess][(sentence, token)])
            else:
                feat = feature_generator.vw_from_title(guess, question.get_text(sentence, token))
            yield sentence, token, guess, feat


def instantiate_feature(feature_name, questions, deep_data="data/deep"):
    """
    @param feature_name: The feature to instantiate
    @param questions: question database
    """

    feature = None
    print("Loading feature %s ..." % feature_name)
    if feature_name == "ir":
        feature = IrExtractor(MIN_APPEARANCES)
    elif feature_name == "lm":
        feature = LanguageModel(data_path('data/lm.txt'))
    elif feature_name == "deep":
        print("from %s" % deep_data)
        page_dict = {}
        for page in questions.get_all_pages():
            page_dict[page.lower().replace(' ', '_')] = page
        feature = DeepExtractor(
            "data/deep/classifier",
            "data/deep/params",
            "data/deep/vocab",
            "data/common/ners",
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


def stream_guesses(text: str, deep_feature: DeepExtractor, features: Dict[str, FeatureExtractor]):
    guesses = deep_feature.text_guess([text])
    output = ''
    for name in FEATURE_NAMES:
        if name == 'deep':
            pass
        else:
            pass


def spark_stream(sc: SparkContext):
    ssc = StreamingContext(sc, 1)
    questions = ssc.socketTextStream('localhost', 9999)


def spark_batch(sc: SparkContext, feature_names, question_db, guess_db, granularity='sentence'):
    sql_context = SQLContext(sc)
    question_db = QuestionDatabase(question_db)

    print("Loading Questions")
    questions = question_db.guess_questions()

    print("Loading Guesses")
    guess_list = GuessList(guess_db)
    guess_lookup = guess_list.all_guesses(allow_train=True)

    if 'deep' in feature_names:
        b_guess_cache = sc.broadcast(guess_list.deep_guess_cache())
    else:
        b_guess_cache = sc.broadcast(None)

    print("Loading tasks")
    tasks = []
    for q in questions:
        #if 'deep' in feature_names:
        #    cache = {}
        #    guess_set = set()
        #    for st_guesses in guess_lookup[q.qnum].values():
        #        guess_set = guess_set.union(st_guesses)
        #    for g in guess_set:
        #        cache[g] = guess_cache[q.qnum][g]
        #else:
        #    cache = None
        tasks.append(Task(q, guess_lookup[q.qnum]))
    shuffle(tasks)
    print("Number of tasks: {0}".format(len(tasks)))

    features = {name: instantiate_feature(name, question_db) for name in feature_names}

    b_features = sc.broadcast(features)

    def f_eval(x):
        return evaluate_feature_question(x, b_features, b_guess_cache, granularity)

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
            filename = '/home/ubuntu/qb/data/features/{0}/{1}.{2}.parquet'\
                .format(fold, granularity, name)

            feature_df_with_fold.filter('feature_name = "{0}"'.format(name))\
                .write.save(filename, mode='overwrite')
        feature_df_with_fold.unpersist()
    print("Computation Completed, stopping Spark")
    sc.stop()


def evaluate_feature_question(task, b_features, b_guess_cache, granularity):
    features = b_features.value
    question = task.question
    result = []
    for feature_name in features:
        feature_generator = features[feature_name]
        cache = b_guess_cache.value[question.qnum]
        for sentence, token, guess, feat in feature_lines(
                question, cache, task.guesses, granularity, feature_generator):
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


def create_guesses(processes=cpu_count()):
    q_db = QuestionDatabase(QB_QUESTION_DB)
    guess_list = GuessList(QB_GUESS_DB)

    deep_feature = instantiate_feature('deep', q_db)
    questions = q_db.guess_questions()
    tasks = []
    for q in questions:
        tasks.append((q, deep_feature))

    with Pool(processes=processes) as pool:
        question_guesses = pool.map(parallel_generate_guesses, tasks)
        for qnum, fold, guesses in question_guesses:
            guess_list.save_guesses('deep', qnum, fold, guesses)

    guess_list.create_indexes()
