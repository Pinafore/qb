from collections import Counter
from math import log

from fuzzywuzzy import fuzz
import dlib

from qanta import logging
from qanta.datasets.quiz_bowl import QuestionDatabase
from qanta.util.constants import ENGLISH_STOP_WORDS

logger = logging.get(__name__)

class RankingExample:
    def __init__(self, row, vector, correct, answer, text, new):
        self.row = row
        self.vector = vector
        self.is_correct = correct
        self.answer = answer
        self.new = new
        self.text = text

class ExampleGenerator:
    """
    Generate the X value for our training data
    """

    def __init__(self):
        self._feat_index = 0
        self._feat_names = {}
        self._feat = []

    def add_feature(self, feat):
        self._feat.append(feat)
        for ii in feat.feature_names:
            self._feat_index += 1
            name = "%s:%s" % (feat.name, ii)
            logger.info("Adding feature %s" % name)
            assert name not in self._feat_names
            self._feat_names[name] = self._feat_index

    def __call__(self, question, guess):
        v = dlib.sparse_vector()
        for feat_type in self._feat:
            extracted = feat_type(question, guess)
            for feat_val in extracted:
                v.append(dlib.pair(self._feat_names[feat_val],
                                   extracted[feat_val]))
        return v

class Feature:
    def __init__(self):
        self.feature_names = [""]

    def __len__(self):
        return len(self.feature_names)

class GuessFrequency(Feature):
    def __init__(self, all_questions, fold="guesstrain"):
        Feature.__init__(self)
        self.name = "guess"
        self.feature_names = set(["logcount"])
        self._count = Counter(x.page for x in all_questions.values()
                              if x.fold == fold)
        self._most_freq = {}

        for ii, cc in self._count.most_common(500):
            self.feature_names.add(ii)

    def __call__(self, text, title):
        """

        @param title: The guess row_iterator
        @param text: Question text
        """
        d = {"guess:logcount": log(self._count[title["guess"]] + 1)}
        if title["guess"] in self.feature_names:
            d["guess:%s" % title["guess"]] = 1.0
        return d

class IrScore(Feature):
    def __init__(self):
        Feature.__init__(self)
        self.feature_names = ["score"]
        self.name = "ir"

    def __call__(self, text, guess):
        return {"ir:score": guess["score"]}

class AnswerPresent(Feature):
    def __call__(self, text, title):
        Feature.__init__(self)
        d = {}
        if "(" in title:
            title = title[:title.find("(")].strip()
        val = fuzz.partial_ratio(title, text)
        d["ap:raw"] = log(val + 1)
        d["ap:length"] = log(val * len(title) / 100. + 1)

        longest_match = 1
        for ii in title.split():
            if ii.lower() in ENGLISH_STOP_WORDS:
                continue
            longest_match = max(longest_match, len(ii) if ii in text else 0)
        d["ap:longest"] = log(longest_match)

        return d

    def __init__(self):
        Feature.__init__(self)
        self.feature_names = ["raw", "length", "longest"]
        self.name = "ap"

class RegexpFeature(Feature):
    def __init__(self, question_pattern, guess_pattern):
        Feature.__init__(self)
        self._q = question_pattern
        self._a = guess_pattern

    def __call__(self, question, guess):
        None

class Reranker:
    def __init__(self):
        self._ranker = None

    @staticmethod
    def row_iterator(example_generator, questions, guesses):
        missing_questions = set()
        last_query = None
        for row, gg in guesses.iterrows():
            sent = gg["sentence"]
            tok = gg["token"]
            qnum = gg["qnum"]

            if not qnum in questions:
                if gg["qnum"] not in missing_questions:
                    logger.info("Missing question %s" % gg["qnum"])
                missing_questions.add(gg["qnum"])
                yield RankingExample(row, None, None, None, None, None)
                continue

            query = (qnum, sent, tok)
            if last_query != query:
                data = dlib.sparse_ranking_pair()
                text = questions[qnum].get_text(sent, tok)
                answer = questions[qnum].page
                new_question = True
            else:
                new_question = False

            guess = gg["guess"]
            yield RankingExample(row,
                                 example_generator(text, gg),
                                 guess==answer, answer, text,
                                 new_question)
            last_query = query

    @staticmethod
    def create_train(example_generator, questions, guesses):
        queries = dlib.sparse_ranking_pairs()
        num_examples = 0

        for ex in Reranker.row_iterator(example_generator,
                                        questions, guesses):
            # Skip bad rows
            if ex.vector is None:
                continue

            if ex.new:
                if num_examples > 0 and has_correct:
                    queries.append(data)

                data = dlib.sparse_ranking_pair()
                has_correct = False

            if ex.is_correct:
                has_correct = True
                data.relevant.append(ex.vector)
            else:
                data.nonrelevant.append(ex.vector)
            num_examples += 1

        queries.append(data)
        return queries

    def score_one(self, example):
        return self._ranker(example)

    def cv_for_c(self, data):
        trainer = dlib.svm_rank_trainer_sparse()
        best_score = 0
        best_c = 0.0
        for cc in [0.1, 1.0, 5.0, 7.5, 10, 12.5, 20, 50]:
            trainer.c = cc
            res = dlib.cross_validate_ranking_trainer(trainer, data, 4)
            logger.info("%f (c=%f)" % (res.ranking_accuracy, cc))
            if res.ranking_accuracy > best_score:
                best_score = res.ranking_accuracy
                best_c = cc
        return best_c

    def train_svm(self, data, c_val=None):
        trainer = dlib.svm_rank_trainer_sparse()
        if c_val:
            trainer.c = c_val
        self._ranker = trainer.train(data)

    def predict(self, guesses):
        None

if __name__ == "__main__":
    import pickle
    import argparse

    parser = argparse.ArgumentParser(description='Learn reranker')

    parser.add_argument('--db', type=str,
                        default='data/internal/2017_05_23.db')
    parser.add_argument('--train_fold', default="guessdev", type=str)
    parser.add_argument('--eval_fold', default="buzzerdev", type=str)
    parser.add_argument('--eval_output', default="eval.csv", type=str)
    parser.add_argument('--eval_rows', default=-1, type=int)
    flags = parser.parse_args()

    train_guess = pickle.load(open("output/guesser/qanta.guesser.elasticsearch.ElasticSearchGuesser/guesses_%s.pickle" % flags.train_fold, 'rb'))

    test_guess = pickle.load(open("output/guesser/qanta.guesser.elasticsearch.ElasticSearchGuesser/guesses_%s.pickle" % flags.eval_fold, 'rb'))


    qdb = QuestionDatabase(flags.db, load_expo=False)
    questions = qdb.all_questions()

    ex_gen = ExampleGenerator()
    ex_gen.add_feature(GuessFrequency(questions))
    ex_gen.add_feature(IrScore())

    train = Reranker.create_train(ex_gen, questions, train_guess)
    ranker = Reranker()
    c = ranker.cv_for_c(train)
    ranker.train_svm(train, c)
    print("Done training (best C=%f)" % c)

    scores = []
    for ex in Reranker.row_iterator(ex_gen, questions, test_guess):
        if ex.row % 10000 == 0:
            logger.info("Generating predictions for example %f percent" %
                        (float(ex.row) / len(test_guess)))
            logger.info(str(ex.vector)[:80])
        if flags.eval_rows > 0 and ex.row > flags.eval_rows:
            break

        if ex.vector is None:
            val = float("NaN")
        else:
            val = ranker.score_one(ex.vector)
        scores.append({"rerank": val, "text": ex.text, "answer": ex.answer})

    with open(flags.eval_output, 'w') as out:
        from csv import DictWriter
        outfile = DictWriter(out, fieldnames=scores[0].keys())
        outfile.writeheader()
        for ii in scores:
            outfile.writerow(ii)
