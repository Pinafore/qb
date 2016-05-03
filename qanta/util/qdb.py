from typing import List, Dict
import sqlite3
import random
import string
from collections import defaultdict, OrderedDict, Counter
import re

from unidecode import unidecode
from whoosh.collectors import TimeLimitCollector, TimeLimit
from qanta.util.constants import MIN_APPEARANCES
from whoosh import qparser
from whoosh.qparser import QueryParser
from fuzzywuzzy import fuzz
from extractors.ir import IrIndex


punc = set(string.punctuation)
paren = re.compile("\\([^\\)]*\\)")


class ClosestQuestion:
    def __init__(self, index_location):
        from whoosh.fields import Schema, TEXT, STORED
        from whoosh.index import create_in, open_dir

        self.schema = Schema(id=STORED, text=TEXT)

        self.index = create_in(index_location, self.schema)
        self.index = open_dir(index_location)
        self.writer = self.index.writer()
        self.raw = {}

        self.parser = None

    def add_question(self, qid, question):
        self.writer.add_document(id=qid, text=question)
        self.raw[qid] = question.lower()

    def finalize(self):
        self.writer.commit()

    def find_closest(self, raw_query, threshold=50):
        """
        Returns the best score of similarity
        """
        if self.parser is None:
            og = qparser.OrGroup.factory(0.9)
            self.parser = QueryParser("text", schema=self.schema,
                                      group=og)

        query_text, _ = IrIndex.prepare_query(raw_query.lower())
        print("Query: %s" % query_text)
        query = self.parser.parse(query_text)
        print("-------------")
        closest_question = -1
        with self.index.searcher() as s:
            c = s.collector(limit=10)
            tlc = TimeLimitCollector(c, timelimit=5)
            try:
                s.search_with_collector(query, tlc)
            except TimeLimit:
                pass
            try:
                results = tlc.results()
            except TimeLimit:
                print("Time limit reached!")
                return -1

            print(results[0]['id'],
                  self.raw[results[0]['id']][:50])
            similarity = fuzz.ratio(self.raw[results[0]['id']],
                                    raw_query.lower())
            if similarity > threshold:
                closest_question = results[0]['id']
                print("Old!", closest_question, similarity)
            else:
                print("NEW!  %f" % similarity)
        print("-------------")
        return closest_question


class Question:
    def __init__(self, qnum, answer, category, naqt,
                 tournaments, page, ans_type, fold, gender):
        self.qnum = qnum
        self.answer = answer
        self.category = category
        self.naqt = naqt
        self.tournaments = tournaments
        self.page = page
        self.ans_type = ans_type
        self.fold = fold
        self.gender = gender
        self.text = {}
        self._last_query = None
        self._cache_query = ""

    def raw_words(self):
        """
        Return a list of all words, removing all punctuation and normalizing
        words
        """
        for ii in sorted(self.text):
            for jj in self.split_and_remove_punc(self.text[ii]):
                yield jj

    @staticmethod
    def split_and_remove_punc(text):
        for ii in text.split():
            word = "".join(x for x in unidecode(ii.lower()) if x not in punc)
            if word:
                yield word

    def partials(self, word_skip=-1):
        assert(isinstance(word_skip, int)), "Needs an integer %i" % word_skip
        for i in sorted(self.text):
            previous = [self.text[x] for x in sorted(self.text) if x < i]

            # TODO(jbg): Test to make sure this gives individual words
            # correctly if word_skip > 0
            if word_skip > 0:
                words = self.text[i].split()
                for j in range(word_skip, len(words), word_skip):
                    yield i, j, previous + [" ".join(words[:j])]

            yield i + 1, 0, [self.text[x] for x in sorted(self.text) if x <= i]

    def text_lines(self):
        d = {}
        d["id"] = self.qnum
        d["answer"] = unidecode(self.page)
        for ii in sorted(self.text):
            d["sent"] = ii
            d["text"] = unidecode(self.text[ii])
            yield d

    def get_text(self, sentence, token):
        if self._last_query != (sentence, token):
            self._last_query = (sentence, token)
            previous = ""
            for ii in range(sentence):
                previous += self.text.get(ii, "")
            if token > 0:
                previous += " ".join(self.text[sentence].split()[:token])
            self._cached_query = previous
        return self._cached_query

    def add_text(self, sent, text):
        self.text[sent] = text

    def random_text(self):
        cutoff = random.randint(0, max(self.text))
        return unidecode("\t".join(self.text[x] for x in sorted(self.text)
                                   if x <= cutoff))

    def flatten_text(self):
        return unidecode("\t".join(self.text[x] for x in sorted(self.text)))


class QuestionDatabase:
    def __init__(self, location):
        self._conn = sqlite3.connect(location)

    def query(self, command, arguments, text=True):
        questions = {}
        c = self._conn.cursor()
        command = 'select id, page, category, answer, ' + \
            'tournament, type, naqt, fold, gender ' + command
        c.execute(command, arguments)

        for qq, pp, cc, aa, tt, kk, nn, ff, gg in c:
            questions[qq] = Question(qq, aa, cc, nn, tt, pp, kk, ff, gg)

        if text:
            for ii in questions:
                command = 'select sent, raw from text where question=? order by sent asc'
                c.execute(command, (ii, ))
                for ss, rr in c:
                    questions[ii].add_text(ss, rr)

        return questions

    def all_questions(self):
        return self.query("FROM questions", ())

    def unmatched_answers(self, ids_to_exclude=None):
        """
        Return a dictionary with the most unmatched pages
        """

        if ids_to_exclude is None:
            ids_to_exclude = set()

        c = self._conn.cursor()
        command = 'select answer, id as cnt from questions ' + \
            'where page == ""'
        c.execute(command)

        answers = defaultdict(dict)
        for aa, qid in c:
            normalized = aa.lower()
            normalized = normalized.replace("_", "")
            if qid not in ids_to_exclude:
                answers[normalized][aa] = answers[normalized].get(aa, 0) + 1
        return answers

    def answer_map(self, normalization=lambda x: x):
        c = self._conn.cursor()
        command = 'select answer, page from questions ' + \
            'where page != ""'
        c.execute(command)

        d = defaultdict(Counter)
        for aa, pp in c:
            d[normalization(aa)][pp] += 1

        return d

    def questions_by_answer(self, answer):
        questions = self.query('from questions where answer == ?', (answer,))

        for ii in questions:
            yield questions[ii]

    def questions_with_pages(self) -> Dict[str, List[Question]]:
        page_map = OrderedDict()

        questions = self.query('from questions where page != ""', ()).values()

        for row in sorted(questions, key=lambda x: x.answer):
            if row.page not in page_map:
                page_map[row.page] = []
            page_map[row.page].append(row)
        return page_map

    def questions_by_tournament(self, tournament):
        return self.query('from questions where tournament like ?', ('%%%s%%' % tournament, ))

    def prune_text(self):
        """
        Remove sentences that do not have an entry in the database
        """

        c = self._conn.cursor()
        command = 'select id from questions group by id'
        c.execute(command)
        questions = set(x for x in c)

        c = self._conn.cursor()
        command = 'select question from text group by question'
        c.execute(command)
        text = set(x for x in c)

        orphans = text - questions

        c = self._conn.cursor()
        for ii in orphans:
            command = 'delete from text where question=%i' % ii
            c.execute(command)
        print("Keeping %i Pruning %i" % (len(questions - orphans),
                                         len(orphans)))
        self._conn.commit()

    def page_by_count(self, min_count=1, exclude_test=False):
        """
        Return all answers that appear at least the specified number
        of times in a category.
        """
        c = self._conn.cursor()
        if exclude_test:
            command = 'select page, count(*) as num from questions where ' + \
                      'page != "" and fold != "test" and fold != "devtest"' + \
                      'group by page order by num desc'
        else:
            command = 'select page, count(*) as num from questions where ' + \
                      'page != "" ' + \
                      'group by page order by num desc'
        c.execute(command)

        for aa, nn in c:
            if nn < min_count:
                continue
            else:
                yield aa

    def get_all_pages(self):
        c = self._conn.cursor()
        c.execute('select page from questions where page != "" group by page')
        for cc in c:
            yield cc[0]

    def majority_frequency(self, page, column):
        """
        Given a page, look up the majority value for a column and its frequency
        """

        c = self._conn.cursor()
        c.execute('SELECT ?, count(*) AS cnt FROM questions WHERE page=? ' +
                  'GROUP BY ? ORDER BY cnt DESC', (column, page, column, ))
        total = 0
        majority = None
        majority_count = 0
        for pp, cc in c:
            print(pp, cc)
            if majority is None:
                majority = pp
                majority_count = cc
            total += cc

        if total > 0:
            return majority, float(majority_count) / float(total)
        else:
            return majority, -1

    def all_answers(self):
        """
        Return a lookup from IDs to pages
        """
        answers = {}
        c = self._conn.cursor()
        command = "select id, page from questions where page != ''"
        c.execute(command)

        for qid, page in c:
            answers[int(qid)] = page
        return answers

    def set_answer_page(self, qid, page, ans_type):
        query = "UPDATE questions SET page=?, type=? WHERE id=?"
        c = self._conn.cursor()
        c.execute(query, (page, ans_type, qid))
        self._conn.commit()


def main():
    db = QuestionDatabase("data/questions.db")
    for ii in db.page_by_count(MIN_APPEARANCES):
        print(unidecode(ii))


if __name__ == "__main__":
    main()
