import numpy as np
from collections import defaultdict, namedtuple
import sqlite3
from typing import Dict, Tuple, Set, Any
from functional import seq

from qanta.util.constants import NEG_INF
from qanta.util.environment import QB_QUESTION_DB
from qanta.util import qdb


Guess = namedtuple('Guess',
                   ['fold', 'question', 'sentence', 'token', 'page', 'guesser', 'feature', 'score'])


class GuessList:
    def __init__(self, db_path):
        # Create the database structure if it doesn't exist
        self.db_path = db_path
        self.db_structure(db_path)
        self._cursor_fail = False
        self._conn = sqlite3.connect(db_path)
        self._stats = {}

    def _cursor(self):
        try:
            return self._conn.cursor()
        except sqlite3.ProgrammingError as e:
            if not self._cursor_fail:
                self._cursor_fail = True
                self._conn = sqlite3.connect(self.db_path)
                return self._conn.cursor()
            else:
                raise sqlite3.ProgrammingError(e)

    def db_structure(self, db_path: str) -> None:
        """
        Creates the database if it does not exist. The table has the following columns.

        fold -> str: which fold from train/test/dev
        question -> int: which question
        sentence -> int: how many sentences have been seen to generate guess
        token -> int: how many tokens have been seen to generate guess
        page -> str: page which is unique answer guess
        guesser -> str: set to "deep"
        feature -> int: unused
        score -> float: score from deep classifier

        :param db_path: path to database
        :return: None
        """
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        sql = 'CREATE TABLE IF NOT EXISTS guesses (' + \
            'fold TEXT, question INTEGER, sentence INTEGER, token INTEGER, page TEXT,' + \
            ' guesser TEXT, feature TEXT, score NUMERIC, PRIMARY KEY ' + \
            '(fold, question, sentence, token, page, guesser, feature));'
        c.execute(sql)
        conn.commit()

    def number_guesses(self, question: Any, guesser: str) -> int:
        query = 'SELECT COUNT(*) FROM guesses WHERE question=? AND guesser=?;'
        c = self._cursor()
        c.execute(query, (question.qnum, guesser,))
        for count, in c:
            return count
        return 0

    def guesses_for_question(self, question) -> Dict[Tuple[int, int], Set[str]]:
        """
        Returns a list of guesses for a given question.
        :param question:
        :return:
        """
        query = 'SELECT sentence, token, page FROM guesses WHERE question=?;'
        c = self._cursor()
        c.execute(query, (question.qnum,))

        guesses = defaultdict(set)
        for sentence, token, page in c:
            guesses[(sentence, token)].add(page)
        return guesses

    def all_guesses(self, allow_train=False) -> Dict[int, Dict[Tuple[int, int], Set[str]]]:
        if allow_train:
            query = 'SELECT question, sentence, token, page FROM guesses'
        else:
            query = 'SELECT question, sentence, token, page FROM guesses where fold != "train"'
        c = self._cursor()
        c.execute(query)
        guesses = {}
        for question, sentence, token, page in c:
            if question not in guesses:
                guesses[question] = defaultdict(set)
            guesses[question][(sentence, token)].add(page)
        return guesses

    def check_recall(self):
        c = self._cursor()
        print("Loading questions and guesses")
        raw_questions = seq(qdb.QuestionDatabase(QB_QUESTION_DB).all_questions().values())
        guesses = seq(list(
            c.execute('select * from guesses where guesser="deep" and fold = "devtest"')))\
            .map(lambda g: Guess(*g)).cache()

        positions = guesses.map(lambda g: (g.question, g.sentence)) \
            .reduce_by_key(max).to_dict()

        guess_lookup = guesses.filter(lambda g: g.sentence == positions[g.question]) \
            .group_by(lambda x: x.question) \
            .map(lambda g: (g[0], seq(g[1]).map(lambda x: x.page).set())).to_dict()

        questions = raw_questions.\
            filter(lambda q: q.qnum in guess_lookup and q.fold != 'train').cache()

        correct = 0
        total = 0
        wrong = []

        print("Computing DAN recall")
        for q in questions:
            if q.page in guess_lookup[q.qnum]:
                correct += 1
            else:
                wrong.append(q)
            total += 1
        return correct / total, total, wrong

    def guesser_statistics(self, guesser, feature, limit=5000):
        """
        Return the mean and variance of a guesser's scores.
        """

        if limit > 0:
            query = 'SELECT score FROM guesses WHERE guesser=? AND feature=? AND score>0 LIMIT %i;' % limit
        else:
            query = 'SELECT score FROM guesses WHERE guesser=? AND feature=? AND score>0;'
        c = self._cursor()
        c.execute(query, (guesser, feature,))

        # TODO(jbg): Is there a way of computing this without casting to list?
        values = list(x[0] for x in c if x[0] > NEG_INF)

        return np.mean(values), np.var(values)

    def get_guesses(self, guesser, question):
        query = 'SELECT sentence, token, page, feature, score ' + \
            'FROM guesses WHERE question=? AND guesser=?;'
        c = self._cursor()
        c.execute(query, (question.qnum, guesser,))

        guesses = defaultdict(dict)
        for ss, tt, pp, ff, vv in c:
            if pp not in guesses[(ss, tt)]:
                guesses[(ss, tt)][pp] = {}
            guesses[(ss, tt)][pp][ff] = vv
        return guesses

    def deep_guess_cache(self):
        query = 'SELECT question, sentence, token, page, feature, score FROM guesses WHERE guesser="deep"'
        c = self._cursor()
        c.execute(query)
        guesses = {}

        for question, sentence, token, page, feature, score in c:
            if question not in guesses:
                guesses[question] = defaultdict(dict)
            if page not in guesses[question][(sentence, token)]:
                guesses[question][page][(sentence, token)] = {}
            guesses[question][page][(sentence, token)][feature] = score
        return guesses

    def add_guesses(self, guesser, question, fold, guesses):
        c = self._cursor()
        # Add in the new guesses
        query = 'INSERT INTO guesses' + \
            '(fold, question, sentence, token, page, guesser, score, feature) ' + \
            'VALUES(?, ?, ?, ?, ?, ?, ?, ?);'
        for ss, tt in guesses:
            for gg in guesses[(ss, tt)]:
                for feat, val in guesses[(ss, tt)][gg].items():
                    c.execute(query, (fold, question, ss, tt, gg, guesser, val, feat))
        self._conn.commit()
