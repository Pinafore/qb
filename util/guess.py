import numpy as np
from collections import defaultdict
import sqlite3


NEG_INF = float('-inf')


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
            'fold TEXT, question INTEGER, sentence INTEGER, token INTEGER, page TEXT,' + \
            ' guesser TEXT, feature TEXT, score NUMERIC, PRIMARY KEY ' + \
            '(fold, question, sentence, token, page, guesser, feature));'
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
        query = 'SELECT sentence, token, page FROM guesses WHERE question=?;'
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
        values = list(x[0] for x in c if x[0] > NEG_INF)

        return np.mean(values), np.var(values)

    def get_guesses(self, guesser, question):
        query = 'SELECT sentence, token, page, feature, score ' + \
            'FROM guesses WHERE question=? AND guesser=?;'
        c = self._conn.cursor()
        # print(query, question.qnum, guesser,)
        c.execute(query, (question.qnum, guesser,))

        guesses = defaultdict(dict)
        for ss, tt, pp, ff, vv in c:
            if pp not in guesses[(ss, tt)]:
                guesses[(ss, tt)][pp] = {}
            guesses[(ss, tt)][pp][ff] = vv
        return guesses

    def add_guesses(self, guesser, question, fold, guesses):
        # Remove the old guesses
        query = 'DELETE FROM guesses WHERE question=? AND guesser=?;'
        c = self._conn.cursor()
        c.execute(query, (question, guesser,))

        # Add in the new guesses
        query = 'INSERT INTO guesses' + \
            '(fold, question, sentence, token, page, guesser, score, feature) ' + \
            'VALUES(?, ?, ?, ?, ?, ?, ?, ?);'
        for ss, tt in guesses:
            for gg in guesses[(ss, tt)]:
                for feat, val in guesses[(ss, tt)][gg].items():
                    c.execute(query,
                              (fold, question, ss, tt, gg,
                               guesser, val, feat))
        self._conn.commit()