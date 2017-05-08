from typing import List, Dict, Iterable, Tuple
import sqlite3
from collections import defaultdict, OrderedDict, Counter
import re

from functional import seq

from qanta import logging
from qanta.preprocess import format_guess
from qanta.datasets.abstract import AbstractDataset, TrainingData, QuestionText, Answer
from qanta.util.environment import QB_QUESTION_DB
from qanta.util.constants import PUNCTUATION
from qanta.config import conf

kPAREN = re.compile(r'\([^)]*\)')
kBRACKET = re.compile(r'\[[^)]*\]')
kMULT_SPACE = re.compile(r'\s+')
kANGLE = re.compile(r'<[^>]*>')

log = logging.get(__name__)


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

    def __repr__(self):
        return '<Question qnum={} page="{}" text="{}...">'.format(
            self.qnum,
            self.page,
            self.flatten_text()[0:20]
        )

    def raw_words(self):
        """
        Return a list of all words, removing all punctuation and normalizing
        words
        """
        for i in sorted(self.text):
            for j in self.split_and_remove_punc(self.text[i]):
                yield j

    @staticmethod
    def split_and_remove_punc(text):
        for i in text.split():
            word = "".join(x for x in i.lower() if x not in PUNCTUATION)
            if word:
                yield word

    def partials(self, word_skip=-1):
        for i in sorted(self.text):
            previous = [self.text[x] for x in sorted(self.text) if x < i]
            if word_skip > 0:
                words = self.text[i].split()
                for j in range(word_skip, len(words), word_skip):
                    yield i, j, previous + [" ".join(words[:j])]

            yield i + 1, 0, [self.text[x] for x in sorted(self.text) if x <= i]

    def text_lines(self):
        d = {"id": self.qnum, "answer": self.page}
        for i in sorted(self.text):
            d["sent"] = i
            d["text"] = self.text[i]
            yield d

    def get_text(self, sentence, token):
        text = ""
        for i in range(sentence):
            text += self.text.get(i, "")
        if token > 0:
            text += " ".join(self.text.get(sentence, "").split()[:token])
        return text

    def add_text(self, sent, text):
        self.text[sent] = text

    def flatten_text(self):
        return " ".join(self.text[x] for x in sorted(self.text))

    def to_example(self) -> Tuple[List[QuestionText], Answer, Dict]:
        sentence_list = [self.text[i] for i in range(len(self.text))]
        properties = {
            'ans_type': self.ans_type,
            'category': self.category.split(':')[0],
            'gender': self.gender
        }
        return sentence_list, self.page, properties


class QuestionDatabase:
    def __init__(self, location=QB_QUESTION_DB):
        self._conn = sqlite3.connect(location)

    def query(self, command: str, arguments) -> Dict[str, Question]:
        questions = {}
        c = self._conn.cursor()
        command = 'select id, page, category, answer, ' + \
            'tournament, type, naqt, fold, gender ' + command
        c.execute(command, arguments)

        for qnum, page, category, answer, tournaments, ans_type, naqt, fold, gender in c:
            questions[qnum] = Question(qnum, answer, category, naqt, tournaments, page, ans_type,
                                       fold, gender)

        for qnum in questions:
            command = 'select sent, raw from text where question=? order by sent asc'
            c.execute(command, (qnum, ))
            for sentence, text in c:
                questions[qnum].add_text(sentence, text)

        return questions

    def all_questions(self, unfiltered=False):
        if unfiltered:
            return self.query('FROM questions', ())
        else:
            return self.query('FROM questions where page != ""', ())

    def answer_map(self):
        c = self._conn.cursor()
        command = 'select answer, page from questions ' + \
            'where page != ""'
        c.execute(command)

        d = defaultdict(Counter)
        for answer, page in c:
            d[answer][page] += 1

        return d

    def normalize_answer(self, answer):
        answer = answer.lower().replace("_ ", " ").replace(" _", " ").replace("_", "")
        answer = answer.replace("{", "").replace("}", "")
        answer = kPAREN.sub('', answer)
        answer = kBRACKET.sub('', answer)
        answer = kANGLE.sub('', answer)
        answer = kMULT_SPACE.sub(' ', answer)
        answer = " ".join(Question.split_and_remove_punc(answer))
        return answer

    def normalized_answers(self):
        """
        Return a dictionary with the most unmatched pages
        """

        c = self._conn.cursor()
        command = 'select answer, page from questions '
        c.execute(command)

        answers = defaultdict(list)
        for aa, page in c:
            normalized = self.normalize_answer(aa)
            answers[normalized].append((aa, page))
        return answers

    def questions_by_answer(self, answer):
        questions = self.query('from questions where answer == ?', (answer,))

        for ii in questions:
            yield questions[ii]

    def questions_with_pages(self, normalize_titles=False) -> Dict[str, List[Question]]:
        page_map = OrderedDict()

        questions = self.query('from questions where page != ""', ()).values()

        for row in sorted(questions, key=lambda x: x.answer):
            if normalize_titles:
                page = format_guess(row.page)
            else:
                page = row.page
            if row.page not in page_map:
                page_map[page] = []
            page_map[page].append(row)
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
        log.info("Keeping %i Pruning %i" % (len(questions - orphans), len(orphans)))
        self._conn.commit()

    def page_by_count(self, min_count: int, exclude_test: bool):
        """
        Return all answers that appear at least the specified number
        of times in a category.
        """
        c = self._conn.cursor()
        if exclude_test:
            command = 'select page, count(*) as num from questions where ' + \
                      'page != "" and fold != "test" and fold != "devtest" and fold != "dev" ' + \
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


class QuizBowlDataset(AbstractDataset):
    def __init__(self, min_class_examples: int, qb_question_db: str=QB_QUESTION_DB):
        """
        Initialize a new quiz bowl data set
        :param min_class_examples: The minimum number of training examples to include an answer class.
        """
        super().__init__()
        self.db = QuestionDatabase(qb_question_db)
        self.min_class_examples = min_class_examples

    def training_data(self, normalize_guess=False) -> TrainingData:
        all_questions = seq(self.db.all_questions().values())
        if conf['guessers_train_on_dev']:
            fold_condition = lambda q: q.fold == 'train' or q.fold == 'dev'
        else:
            fold_condition = lambda q: q.fold == 'train'
        filtered_questions = all_questions\
            .filter(fold_condition)\
            .group_by(lambda q: q.page)\
            .filter(lambda kv: len(kv[1]) >= self.min_class_examples)\
            .flat_map(lambda kv: kv[1])\
            .map(lambda q: q.to_example())
        training_examples = []
        training_answers = []
        training_properties = []
        for example, answer, properties in filtered_questions:
            training_examples.append(example)
            if normalize_guess:
                training_answers.append(format_guess(answer))
            else:
                training_answers.append(answer)
            training_properties.append(properties)

        return training_examples, training_answers, training_properties

    def questions_by_fold(self) -> Dict[str, List[Question]]:
        all_questions = seq(self.db.all_questions().values())
        train_questions = all_questions\
            .filter(lambda q: q.fold == 'train')\
            .group_by(lambda q: q.page)\
            .filter(lambda kv: len(kv[1]) >= self.min_class_examples)\
            .flat_map(lambda kv: kv[1])\
            .list()

        dev_questions = all_questions.filter(lambda q: q.fold == 'dev').list()
        test_questions = all_questions.filter(lambda q: q.fold == 'test').list()
        devtest_questions = all_questions.filter(lambda q: q.fold == 'devtest').list()

        return {
            'train': train_questions,
            'dev': dev_questions,
            'test': test_questions,
            'devtest': devtest_questions
        }

    def questions_in_folds(self, folds: Iterable[str]) -> List[Question]:
        by_fold = self.questions_by_fold()
        questions = []
        for fold in folds:
            questions += by_fold[fold]

        return questions
