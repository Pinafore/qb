from collections import defaultdict, namedtuple
import sqlite3
from typing import Dict, Tuple, Set

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
