from collections import defaultdict, namedtuple
import sqlite3
from typing import Dict, Tuple, Set
from functional import seq

from qanta.util.environment import QB_QUESTION_DB
from qanta.util import qdb

Guess = namedtuple('Guess', ['fold', 'question', 'sentence', 'token', 'page', 'guesser', 'score'])


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
              ' guesser TEXT, score NUMERIC, PRIMARY KEY ' + \
              '(fold, question, sentence, token, page, guesser));'
        c.execute(sql)
        conn.commit()

    def create_indexes(self):
        c = self._cursor()
        sql = 'CREATE INDEX Idx1 ON guesses(fold);'
        c.execute(sql)
        sql = 'CREATE INDEX Idx2 ON guesses(question);'
        c.execute(sql)
        sql = 'CREATE INDEX Idx3 ON guesses(guesser);'
        c.execute(sql)

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
            query = 'SELECT question, sentence, token, page FROM guesses WHERE fold != "train"'
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
            c.execute('SELECT * FROM guesses WHERE guesser="deep" AND fold = "devtest"'))) \
            .map(lambda g: Guess(*g)).cache()

        positions = guesses.map(lambda g: (g.question, g.sentence)) \
            .reduce_by_key(max).to_dict()

        guess_lookup = guesses.filter(lambda g: g.sentence == positions[g.question]) \
            .group_by(lambda x: x.question) \
            .map(lambda g: (g[0], seq(g[1]).map(lambda x: x.page).set())).to_dict()

        questions = raw_questions. \
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

    def get_guesses(self, guesser, question):
        query = 'SELECT sentence, token, page, score ' + \
                'FROM guesses WHERE question=? AND guesser=?;'
        c = self._cursor()
        c.execute(query, (question.qnum, guesser,))

        guesses = defaultdict(dict)
        for sentence, token, page, score in c:
            guesses[(sentence, token)][page] = score
        return guesses

    def deep_guess_cache(self):
        query = 'SELECT question, sentence, token, page, score FROM guesses WHERE guesser="deep"'
        c = self._cursor()
        c.execute(query)
        guesses = {}

        for question, sentence, token, page, score in c:
            if question not in guesses:
                guesses[question] = defaultdict(dict)
            guesses[question][page][(sentence, token)] = score
        return guesses

    def save_guesses(self, guesser, question, fold, guesses):
        rows = []
        for sentence, token in guesses:
            for guess, score in guesses[(sentence, token)].items():
                rows.append((fold, question, sentence, token, guess, guesser, score))

        query = 'INSERT INTO guesses' + \
                '(fold, question, sentence, token, page, guesser, score) ' + \
                'VALUES(?, ?, ?, ?, ?, ?, ?);'
        c = self._cursor()
        c.executemany(query, rows)
        self._conn.commit()
