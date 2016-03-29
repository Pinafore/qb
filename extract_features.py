from collections import defaultdict, namedtuple
import sys
import time
from unidecode import unidecode
from random import shuffle

from pyspark.sql import SQLContext, Row

from util.qdb import QuestionDatabase
from util.guess import GuessList
from util.environment import data_path
from util.constants import FOLDS, MIN_APPEARANCES, FEATURES
from util.build_whoosh import text_iterator
from util.spark_features import SCHEMA

from extractors.label import Labeler
from extractors.ir import IrExtractor
from extractors.text import TextExtractor
from extractors.lm import LanguageModel
from extractors.deep import DeepExtractor
from extractors.classifier import Classifier
from extractors.wikilinks import WikiLinks
from extractors.mentions import Mentions
from extractors.answer_present import AnswerPresent

HAS_GUESSES = set([e.has_guess() for e in [IrExtractor, LanguageModel, TextExtractor, DeepExtractor,
                                           Classifier, AnswerPresent]])

Task = namedtuple('Task', ['page', 'question'])


def feature_lines(question, guess_list, granularity, feature_generator):
    guesses_needed = guess_list.all_guesses(question)

    # Guess we might have already
    # It has the structure:
    # guesses[(sent, token)][page][feat] = value
    guesses_cached = None
    if feature_generator.has_guess():
        guesses_cached = guess_list.get_guesses(feature_generator.name, question)

    for sentence, token in guesses_needed:
        if granularity == "sentence" and token > 0:
            continue

        # Set metadata so the labeler can create ids and weights
        guess_size = len(guesses_needed[(sentence, token)])
        feature_generator.set_metadata(question.page, question.category, question.qnum, sentence,
                                       token, guess_size, question)

        for guess in guesses_needed[(sentence, token)]:
            if guesses_cached is not None and guess in guesses_cached[(sentence, token)]:
                feat = feature_generator.vw_from_score(guesses_cached[(sentence, token)][guess])
            else:
                try:
                    feat = feature_generator.vw_from_title(
                        guess, question.get_text(sentence, token))
                except ValueError:
                    print("Value error!")
                    feat = ""
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
    elif feature_name == "text":
        feature = TextExtractor()
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
            page_dict,
            200
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


def guesses_for_question(question, deep_feature, guess_list=None,
                         word_skip=-1):
    guesses = defaultdict(dict)
    feature_name = 'deep'

    # Find out the guesses that we need for this question
    if guess_list is None or guess_list.number_guesses(question, feature_name) == 0:
        guesses[feature_name] = defaultdict(dict)

    # Gather all the guesses
    for ss, ww, tt in question.partials(word_skip):
        # We have problems at the very start
        if ss == 0 and ww == word_skip:
            continue
        for ff in guesses:
            # print("Query from %s, %s" % (type(tt), tt))
            results = deep_feature.text_guess(tt)
            for gg in results:
                guesses[deep_feature][(ss, ww)][gg] = results[gg]
            # add the correct answer if this is a training document and
            if question.fold == "train" and question.page not in results:
                guesses[feature_name][(ss, ww)][question.page] = \
                  deep_feature.score_one_guess(question.page, tt)

            print(".", end="")
            sys.stdout.flush()

        # Get all of the guesses
        all_guesses = set()
        for gg in guesses[feature_name][(ss, ww)]:
            all_guesses.add(gg)

        # Add missing guesses
        missing = 0
        for gg in [x for x in all_guesses if x not in guesses[feature_name][(ss, ww)]]:
            guesses[feature_name][(ss, ww)][gg] = deep_feature.score_one_guess(gg, tt)
            missing += 1
    return guesses


def spark_execute(sc, feature_names, question_db, guess_db, answer_limit=5, granularity='sentence'):
    sql_context = SQLContext(sc)
    question_db = QuestionDatabase(question_db)

    print("Loading Guesses")
    guess_list = GuessList(guess_db)
    b_guess_list = sc.broadcast(guess_list)

    print("Loading Questions")
    all_questions = question_db.questions_with_pages()
    pages = list(filter(lambda p: len(all_questions[p]) > answer_limit, all_questions.keys()))

    print("Loading tasks")
    tasks = []
    for page in pages:
        for question in filter(lambda q: q.fold != 'train', all_questions[page]):
            tasks.append(Task(page, question))
    shuffle(tasks)
    print("Number of tasks: {0}".format(len(tasks)))

    features = {name: instantiate_feature(name, question_db) for name in feature_names}

    b_features = sc.broadcast(features)
    f_eval = lambda x: evaluate_feature_question(x, b_features, b_guess_list, granularity)

    print("Beginning feature job")
    tasks_rdd = sc.parallelize(tasks)
    feature_rdd = sc.parallelize(feature_names)\
        .cartesian(tasks_rdd).repartition(150 * len(feature_names))\
        .flatMap(f_eval)\
        .cache()

    feature_df = sql_context.createDataFrame(feature_rdd, SCHEMA).repartition(32).cache()
    feature_df.count()
    print("Beginning write job")
    for fold in FOLDS:
        feature_df_with_fold = feature_df.filter('fold = "{0}"'.format(fold)).cache()
        for name in feature_names:
            filename = '/home/ubuntu/output/features/{0}/{1}.{2}.parquet'\
                .format(fold, granularity, name)

            feature_df_with_fold.filter('feature_name = "{0}"'.format(name))\
                .write.save(filename, mode='overwrite')
        feature_df_with_fold.unpersist()
    print("Computation Completed, stopping Spark")
    sc.stop()


def evaluate_feature_question(pair, b_features, b_guess_list, granularity):
    feature_name = pair[0]
    feature_generator = b_features.value[feature_name]
    question = pair[1].question
    result = []
    for sentence, token, guess, feat in feature_lines(
            question, b_guess_list.value, granularity, feature_generator):
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


def main():
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--guesses', default=False, action='store_true',
                        help="Write the guesses")
    parser.add_argument('--label', default=False, action='store_true',
                        help="Write the labels")
    parser.add_argument('--gap', type=int, default=100,
                        help='Gap (in number of tokens) between each guess')
    parser.add_argument('--guess_db', type=str, default='data/guesses.db',
                        help='Where we write/read the guesses')
    parser.add_argument('--question_db', type=str, default='data/questions.db')
    parser.add_argument('--feature', type=str, default='',
                        help="Which feature we write out")
    parser.add_argument("--granularity", type=str,
                        default="sentence")
    parser.add_argument("--limit", type=int, default=-1,
                        help="How many answer to write to feature files")
    parser.add_argument("--ans_limit", type=int, default=5,
                        help="minimum answer limit")

    flags = parser.parse_args()

    print("Loading database from %s" % flags.question_db)
    questions = QuestionDatabase(flags.question_db)
    guess_list = GuessList(flags.guess_db)

    if flags.guesses:
        deep_feature = instantiate_feature('deep', questions)
        print("Guesses deep")

        all_questions = questions.questions_with_pages()

        page_num = 0
        total_pages = sum(1 for x in all_questions if
                          len(all_questions[x]) >= flags.ans_limit)
        for page in all_questions:
            if len(all_questions[page]) < flags.ans_limit:
                continue
            else:
                print("%s\t%i" % (page, len(all_questions[page])))
                question_num = 0
                page_num += 1
                for question in all_questions[page]:
                    # We don't need guesses for train questions
                    if question.fold == "train":
                        continue
                    question_num += 1
                    guesses = guesses_for_question(question, deep_feature, guess_list)

                    # Save the guesses
                    for guesser in guesses:
                        guess_list.add_guesses(guesser, question.qnum, question.fold,
                                               guesses[guesser])
                    print("%i/%i" % (question_num, len(all_questions[page])))

                print("%i(%i) of\t%i\t%s\t" %
                      (page_num, len(all_questions[page]), total_pages, page), end="")

                if 0 < flags.limit < page_num:
                    break

    if flags.feature or flags.label:
        feature_files = {}
        meta = {}
        count = defaultdict(int)

        if flags.feature:
            assert flags.feature in FEATURES, "%s not a feature" % flags.feature
            FEATURES[flags.feature] = instantiate_feature(flags.feature, questions)
            feature_generator = FEATURES[flags.feature]
        else:
            feature_generator = instantiate_feature("label", questions)

        for fold in FOLDS:
            name = feature_generator.name
            filename = ("features/%s/%s.%s.feat" % (fold, flags.granularity, name))
            print("Opening %s for output" % filename)

            feature_files[fold] = open(filename, 'w')
            if flags.label:
                filename = ("features/%s/%s.meta" % (fold, flags.granularity))
            else:
                filename = ("features/%s/%s.meta" % (fold, flags.feature))
            meta[fold] = open(filename, 'w')

        all_questions = questions.questions_with_pages()

        totals = defaultdict(int)
        for page in all_questions:
            for question in all_questions[page]:
                totals[question.fold] += 1
        print("TOTALS")
        print(totals)

        page_count = 0
        feat_lines = 0
        start = time.time()
        max_relevant = sum(1 for x in all_questions if len(all_questions[x]) >= flags.ans_limit)

        for page in all_questions:
            if len(all_questions[page]) >= flags.ans_limit:
                page_count += 1
                if page_count % 50 == 0:
                    print(count)
                    print("Page %i of %i (%s), %f feature lines per sec" %
                          (page_count, max_relevant,
                           feature_generator.name,
                           float(feat_lines) / (time.time() - start)))
                    print(unidecode(page))
                    feat_lines = 0
                    start = time.time()

                for question in all_questions[page]:
                    if question.fold != 'train':
                        count[question.fold] += 1
                        fold_here = question.fold
                        # All the guesses we need to make (on non-train questions)
                        for sentence, token, guess, feat in feature_lines(
                                question, guess_list, flags.granularity, feature_generator):
                            feat_lines += 1
                            if meta:
                                meta[question.fold].write(
                                    "%i\t%i\t%i\t%s\n" % (question.qnum, sentence, token,
                                                          unidecode(guess))
                                )
                            assert feat is not None
                            feature_files[question.fold].write("%s\n" % feat)
                            assert fold_here == question.fold, "%s %s" % (fold_here, question.fold)
                            # print(ss, tt, pp, feat)
                        feature_files[question.fold].flush()

                if 0 < flags.limit < page_count:
                    break


if __name__ == "__main__":
    main()
