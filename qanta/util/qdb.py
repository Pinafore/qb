from typing import List, Dict
import sqlite3
import random
from collections import defaultdict, OrderedDict, Counter

from unidecode import unidecode
from functional import seq

from qanta.util.constants import MIN_APPEARANCES, PUNCTUATION


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
            word = "".join(x for x in unidecode(ii.lower()) if x not in PUNCTUATION)
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

    def flatten_text(self):
        return unidecode(" ".join(self.text[x] for x in sorted(self.text)))


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
        return self.query('FROM questions where page != ""', ())

    def guess_questions(self, appearance_filter=lambda pq: len(pq[1]) >= MIN_APPEARANCES):
        question_pages = self.questions_with_pages()

        dev_questions = seq(question_pages.values()) \
            .flatten() \
            .filter(lambda q: q.fold == 'train' or q.fold == 'dev') \
            .group_by(lambda q: q.page) \
            .filter(appearance_filter) \
            .flat_map(lambda pq: pq[1]) \
            .filter(lambda q: q.fold != 'train')

        test_questions = seq(question_pages.values()) \
            .flatten() \
            .filter(lambda q: q.fold == 'test' or q.fold == 'devtest') \
            .filter(lambda q: q.page != '')

        return (dev_questions + test_questions).list()

    def answer_map(self, normalization=lambda x: x):
        c = self._conn.cursor()
        command = 'select answer, page from questions ' + \
            'where page != ""'
        c.execute(command)

        d = defaultdict(Counter)
        for aa, pp in c:
            d[normalization(aa)][pp] += 1

        return d

    def questions_with_pages(self) -> Dict[str, List[Question]]:
        page_map = OrderedDict()

        questions = self.query('from questions where page != ""', ()).values()

        for row in sorted(questions, key=lambda x: x.answer):
            if row.page not in page_map:
                page_map[row.page] = []
            page_map[row.page].append(row)
        return page_map

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

    def get_all_pages(self, exclude_test=False):
        c = self._conn.cursor()
        if exclude_test:
            c.execute('select distinct page from questions where page != "" and fold != "devtest" and fold != "test"')
        else:
            c.execute('select distinct page from questions where page != ""')
        for result_tuple in c:
            yield result_tuple[0]

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
