from __future__ import print_function
from collections import defaultdict, OrderedDict
import argparse
import sqlite3
import sys
import pickle
from unidecode import unidecode

from numpy import var, mean

from util.qdb import QuestionDatabase
from extractors.ir import IrExtractor
from extractors.text import TextExtractor
from feature_extractor import FeatureExtractor
from extractors.lm import *
from extractors.deep import *
from extractors.classifier import *
from extractors.wikilinks import *
from extractors.title_not_in_qtext import TitleNotinQTextExtractor

kFEATURES = OrderedDict([("ir", None), ("lm", None), ("deep", None),
    ("title_not_in_qtext", None), # ("text", None),
    ("classifier", None), ("wikilinks", None),
    ])

# Add features that actually guess
# TODO: Make this less cumbersome
kHAS_GUESSES = set()
if IrExtractor.has_guess():
    kHAS_GUESSES.add("ir")
if LanguageModel.has_guess():
    kHAS_GUESSES.add("lm")
if TextExtractor.has_guess():
    kHAS_GUESSES.add("text")
if DeepExtractor.has_guess():
    kHAS_GUESSES.add("deep")
if Classifier.has_guess():
    kHAS_GUESSES.add("classifier")
if TitleNotinQTextExtractor.has_guess():
    kHAS_GUESSES.add("title_not_in_qtext")

kGRANULARITIES = ["sentence"]
kFOLDS = ["dev", "devtest", "test"]
kNEGINF = float("-inf")


def feature_lines(qq, guess_list, granularity, feature_generator):
    guesses_needed = guess_list.all_guesses(qq)

    # Guess we might have already
    # It has the structure:
    # guesses[(sent, token)][page][feat] = value
    guesses_cached = defaultdict(dict)
    if feature_generator.has_guess():
        guesses_cached = \
            guess_list.get_guesses(feature_generator.name(), qq)

    for ss, tt in guesses_needed:
        if granularity == "sentence" and tt > 0:
            continue

        # Set metadata so the labeler can create ids and weights
        guess_size = len(guesses_needed[(ss, tt)])
        feature_generator.set_metadata(qq.page, qq.category,
                                       qq.qnum, ss, tt,
                                       guess_size, qq.fold)

        # print("*", qq.qnum, ss, tt, str(guesses_cached[(ss, tt)])[:160])
        for pp in sorted(guesses_needed[(ss, tt)]):
            # Check to see if it's cached
            if pp in guesses_cached[(ss, tt)]:
                # print(guesses_cached[(ss, tt)][pp])
                feat = feature_generator.\
                    vw_from_score(guesses_cached[(ss, tt)][pp])
            else:
                feat = feature_generator.\
                    vw_from_title(pp, qq.get_text(ss, tt))
            # print(pp, feat)
            yield ss, tt, pp, feat


def instantiate_feature(feature_name, questions):
    """
    @param feature_name: The feature to instantiate
    @param questions: question database
    @param first_pass_guess: Is this our first pass generating guesses?  (Used for standardizing IR scores)
    """

    feature = None
    print("Loading feature %s ..." % feature_name)
    if feature_name == "ir":
        feature = IrExtractor()
        for cc in kIR_CUTOFFS:
            wiki_mean = 0.0
            wiki_var = 1.0
            qb_mean = 0.0
            qb_var = 1.0

            feature.add_index("wiki_%i" % cc, "%s_%i" %
                                      ("data/ir/whoosh_wiki", cc),
                                      wiki_mean, wiki_var)
            feature.add_index("qb_%i" % cc, "%s_%i" %
                                      ("data/ir/whoosh_qb", cc),
                                      qb_mean, qb_var)
        if kIR_CATEGORIES:
            categories = questions.column_options("category")
            print("Adding categories %s" % str(categories))
            for cc in categories:
                wiki_mean = 0.0
                wiki_var = 1.0
                qb_mean = 0.0
                qb_var = 1.0

                feature.add_index("wiki_%s" % cc, "%s_%s" %
                                  ("data/ir/whoosh_wiki", cc),
                                  wiki_mean, wiki_var)
                feature.add_index("qb_%s" % cc, "%s_%s" %
                                  ("data/ir/whoosh_qb", cc),
                                  qb_mean, qb_var)
    elif feature_name == "text":
        feature = TextExtractor()
    elif feature_name == "lm":
        feature = pickle.load(open("data/lm.pkl"))
    elif feature_name == "deep":
        page_dict = {}
        for page in questions.get_all_pages():
            page_dict[page.lower().replace(' ', '_')] = page
        feature = DeepExtractor("data/deep/classifier", \
            "data/deep/params", "data/deep/vocab", \
            "data/common/ners", page_dict, 200)
    elif feature_name == "wikilinks":
        feature = WikiLinks()
    elif feature_name == "title_not_in_qtext":
        feature = TitleNotinQTextExtractor(questions)
    elif feature_name == "label":
        feature = Labeler(questions)
    elif feature_name == "classifier":
        feature = Classifier('data/classifier/bigrams.pkl', questions)
    else:
        print("Don't know what to do with %s" % feature_name)
    print("done")
    return feature


def guesses_for_question(qq, features_that_guess, guess_list,
                         word_skip=-1):
    guesses = {}

    # Find out the guesses that we need for this question
    for ff in features_that_guess:
        if guess_list.number_guesses(qq, ff) == 0:
            guesses[ff] = defaultdict(dict)

    # Gather all the guesses
    for ss, ww, tt in qq.partials(word_skip):
        # We have problems at the very start
        if ss == 0 and ww == word_skip:
            continue
        for ff in guesses:
            # print("Query from %s, %s" % (type(tt), tt))
            results = features_that_guess[ff].text_guess(tt)
            for gg in results:
                guesses[ff][(ss, ww)][gg] = results[gg]
            # add the correct answer if this is a training document and
            if qq.fold == "train" and not qq.page in results:
                guesses[ff][(ss, ww)][qq.page] = \
                  features_that_guess[ff].score_one_guess(qq.page, tt)

            print(".", end="")
            sys.stdout.flush()

        # Get all of the guesses
        all_guesses = set()
        for ff in guesses:
            for gg in guesses[ff][(ss, ww)]:
                all_guesses.add(gg)

        # Add missing guesses
        for ff in features_that_guess:
            missing = 0
            for gg in [x for x in all_guesses if not x in
                        guesses[ff][(ss, ww)]]:
                guesses[ff][(ss, ww)][gg] = \
                    features_that_guess[ff].score_one_guess(gg, tt)
                missing += 1
    return guesses

class Labeler(FeatureExtractor):
    def __init__(self, question_db):
        self._correct = None
        self._num_guesses = 0

        all_questions = question_db.questions_with_pages()
        self._counts = {}

        # Get the counts
        for ii in all_questions:
            self._counts[ii] = sum(1 for x in all_questions[ii] if
                                   x.fold == "train")
        # Standardize the scores
        count_mean = mean(self._counts.values())
        count_var = var(self._counts.values())
        for ii in all_questions:
            self._counts[ii] = float(self._counts[ii] - count_mean) / count_var

    def vw_from_title(self, title, query):
        assert self._correct, "Answer not set"
        title = title.replace(":", "").replace("|", "")

        # TODO: Incorporate token position here as well to improve
        # position-based features
        if title == self._correct:
            return "1 '%s |guess %s sent:%0.1f count:%f " % \
                (self._id, unidecode(title).replace(" ", "_"), self._sent,
                 self._counts.get(title, -2))
        else:
            return "-1 %i '%s |guess %s sent:%0.1f count:%f " % \
                (self._num_guesses, self._id,
                 unidecode(title).replace(" ", "_"), self._sent,
                 self._counts.get(title, -2))

    def name(self):
        return "label"


class GuessList:
    def __init__(self, db_path):
        # Create the database structure if it doesn't exist
        self.db_structure(db_path)
        self._conn = sqlite3.connect(db_path)
        self._stats = {}

    def db_structure(self, db_path):
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        sql = 'CREATE TABLE IF NOT EXISTS guesses (' + \
            'question INTEGER, sentence INTEGER, token INTEGER, page TEXT,' + \
            ' guesser TEXT, feature TEXT, score NUMERIC, PRIMARY KEY ' + \
            '(question, sentence, token, page, guesser, feature));'
        c.execute(sql)
        conn.commit()

    def number_guesses(self, question, guesser):
        query = 'SELECT COUNT(*) FROM guesses WHERE question=? AND guesser=?;'
        c = self._conn.cursor()
        c.execute(query, (question.qnum, guesser,))
        for count, in c:
            return count
        return 0

    def all_guesses(self, question):
        query = 'SELECT sentence, token, page  ' + \
            'FROM guesses WHERE question=?;'
        c = self._conn.cursor()
        c.execute(query, (question.qnum,))

        guesses = defaultdict(set)
        for ss, tt, pp in c:
            guesses[(ss, tt)].add(pp)
        if question.page and question.fold == "train":
            for (ss, tt) in guesses:
                guesses[(ss, tt)].add(question.page)
        return guesses

    def check_recall(self, question_list, guesser_list, correct_answer):
        totals = defaultdict(int)
        correct = defaultdict(int)
        c = self._conn.cursor()

        query = 'SELECT count(*) as cnt FROM guesses WHERE guesser=? ' + \
            'AND page=? AND question=?;'
        for gg in guesser_list:
            for qq in question_list:
                if qq.fold == "train":
                    continue

                c.execute(query, (gg, correct_answer, qq.qnum,))
                data = c.fetchone()[0]
                if data != 0:
                    correct[gg] += 1
                totals[gg] += 1

        for gg in guesser_list:
            if totals[gg] > 0:
                yield gg, float(correct[gg]) / float(totals[gg])

    def guesser_statistics(self, guesser, feature, limit=5000):
        """
        Return the mean and variance of a guesser's scores.
        """

        if limit > 0:
            query = 'SELECT score FROM guesses WHERE guesser=? AND feature=? AND score>0 LIMIT %i;' % limit
        else:
            query = 'SELECT score FROM guesses WHERE guesser=? AND feature=? AND score>0;'
        c = self._conn.cursor()
        c.execute(query, (guesser, feature,))

        # TODO(jbg): Is there a way of computing this without casting to list?
        values = list(x[0] for x in c if x[0] > kNEGINF)

        return mean(values), var(values)

    def get_guesses(self, guesser, question):
        query = 'SELECT sentence, token, page, feature, score ' + \
            'FROM guesses WHERE question=? AND guesser=?;'
        c = self._conn.cursor()
        # print(query, question.qnum, guesser,)
        c.execute(query, (question.qnum, guesser,))

        guesses = defaultdict(dict)
        for ss, tt, pp, ff, vv in c:
            if not pp in guesses[(ss, tt)]:
                guesses[(ss, tt)][pp] = {}
            guesses[(ss, tt)][pp][ff] = vv
        return guesses

    def add_guesses(self, guesser, question, guesses):
        # Remove the old guesses
        query = 'DELETE FROM guesses WHERE question=? AND guesser=?;'
        c = self._conn.cursor()
        c.execute(query, (question, guesser,))

        # Add in the new guesses
        query = 'INSERT INTO guesses' + \
            '(question, sentence, token, page, guesser, score, feature) ' + \
            'VALUES(?, ?, ?, ?, ?, ?, ?);'
        for ss, tt in guesses:
            for gg in guesses[(ss, tt)]:
                for feat, val in guesses[(ss, tt)][gg].items():
                    c.execute(query,
                              (question, ss, tt, gg, guesser, val, feat))
        self._conn.commit()

if __name__ == "__main__":
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
        # kFEATURES["ir"] = IrExtractor()
        # for cc in kIR_CUTOFFS:
        #     kFEATURES["ir"].add_index("wiki_%i" % cc, "%s_%i" %
        #                               (flags.whoosh_wiki, cc))
        #     kFEATURES["ir"].add_index("qb_%i" % cc, "%s_%i" %
        #                               (flags.whoosh_qb, cc))
        # if kIR_CATEGORIES:
        #     categories = questions.column_options("category")
        #     print("Adding categories %s" % str(categories))
        #     for cc in categories:
        #         kFEATURES["ir"].add_index("wiki_%s" % cc, "%s_%s" %
        #                                   (flags.whoosh_wiki, cc))
        #         kFEATURES["ir"].add_index("qb_%s" % cc, "%s_%s" %
        #                                   (flags.whoosh_qb, cc))


        kFEATURES["deep"] = instantiate_feature("deep", questions)
        # features_that_guess = set(kFEATURES[x] for x in kHAS_GUESSES)
        features_that_guess = {"deep": kFEATURES["deep"]}
        print("Guesses %s" % "\t".join(x for x in features_that_guess))

        all_questions = questions.questions_with_pages()
        page_num = 0
        for page in all_questions:
            if len(all_questions[page]) < flags.ans_limit:
                continue
            else:
                print("%s\t%i" % (page, len(all_questions[page])))
                question_num = 0
                page_num += 1
                for qq in all_questions[page]:
                    # We don't need guesses for train questions
                    if qq.fold == "train":
                        continue
                    question_num += 1
                    guesses = guesses_for_question(qq, features_that_guess,
                                                   guess_list)

                    # Save the guesses
                    for guesser in guesses:
                        guess_list.add_guesses(guesser, qq.qnum, guesses[guesser])
                    print("%i/%i" % (question_num, len(all_questions[page])))

                print("%i(%i) of\t%i\t%s\t" %
                    (page_num, len(all_questions[page]),
                     len(all_questions), page), end="")

    if flags.feature or flags.label:
        o = {}
        meta = {}
        if flags.feature:
            assert flags.feature in kFEATURES, "%s not a feature" % flags.feature
            kFEATURES[flags.feature] = instantiate_feature(flags.feature,
                                                           questions)
            feature_generator = kFEATURES[flags.feature]
        else:
            feature_generator = instantiate_feature("label", questions)

        for ii in kFOLDS:
            name = feature_generator.name()
            filename = ("features/%s/%s.%s.feat" %
                        (ii, flags.granularity, name))
            print("Opening %s for output" % filename)

            o[ii] = open(filename, 'w')
            if flags.label:
                filename = ("features/%s/%s.meta" %
                                (ii, 'label'))
            else:
                filename = ("features/%s/%s.meta" %
                                (ii, flags.feature))
            meta[ii] = open(filename, 'w')

        all_questions = questions.questions_with_pages()
        page_count = 0
        feat_lines = 0
        start = time.time()
        for page in all_questions:
            if len(all_questions[page]) > flags.ans_limit:
                page_count += 1
                if page_count % 50 == 0:
                    print("Page %i of %i (%s), %f feature lines per sec" %
                          (page_count, len(all_questions),
                           feature_generator.name(),
                           float(feat_lines) / (time.time() - start)))
                    print(unidecode(page))
                    feat_lines = 0
                    start = time.time()

                for qq in all_questions[page]:
                    if qq.fold != 'train':
                        # All the guesses we need to make (on non-train questions)
                        for ss, tt, pp, feat in feature_lines(qq, guess_list,
                                                              flags.granularity,
                                                              feature_generator):
                            feat_lines += 1
                            if meta:
                                meta[qq.fold].write("%i\t%i\t%i\t%s\n" %
                                                    (qq.qnum, ss, tt,
                                                     unidecode(pp)))
                            assert feat is not None
                            o[qq.fold].write("%s\n" % feat)
                            # print(ss, tt, pp, feat)
                        o[qq.fold].flush()

                if flags.limit > 0 and page_count > flags.limit:
                    break
