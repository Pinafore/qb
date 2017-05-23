from typing import List, Dict, Iterable, Tuple
import csv
import os
import sqlite3
from collections import defaultdict, Counter
import re

from functional import seq
import nltk

from qanta import logging
from qanta.datasets.abstract import AbstractDataset, TrainingData, QuestionText, Answer
from qanta.util.environment import QB_QUESTION_DB
from qanta.util import constants as c
from qanta.util.io import file_backed_cache_decorator, safe_path
from qanta.config import conf

kPAREN = re.compile(r'\([^)]*\)')
kBRACKET = re.compile(r'\[[^)]*\]')
kMULT_SPACE = re.compile(r'\s+')
kANGLE = re.compile(r'<[^>]*>')

log = logging.get(__name__)


class Question:
    def __init__(self, qnum, answer, category, naqt, protobowl,
                 tournaments, page, fold):
        self.qnum = qnum
        self.answer = answer
        self.category = category
        self.naqt = naqt
        self.protobowl = protobowl
        self.tournaments = tournaments
        self.page = page
        self.fold = fold
        self.text = {}
        self._last_query = None

    def __repr__(self):
        return '<Question qnum={} page="{}" text="{}...">'.format(
            self.qnum,
            self.page,
            self.flatten_text()[0:20]
        )

    def normalized_answer(self):
        return QuestionDatabase.normalize_answer(self.answer)

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
            word = "".join(x for x in i.lower() if x not in c.PUNCTUATION)
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

    def to_example(self) -> Tuple[List[QuestionText], Answer]:
        sentence_list = [self.text[i] for i in range(len(self.text))]
        return sentence_list, self.page


@file_backed_cache_decorator(safe_path('data/external/preprocess_expo_questions.cache'))
def preprocess_expo_questions(expo_csv: str, database=QB_QUESTION_DB, start_qnum=50000) -> List[Question]:
    """
    This function takes the expo fold and converts it to a list of questions in the same output format as the database.
    
    The start_qnum parameter was determined by looking at the distribution of qnums and finding a range where there are
    no keys. Nonetheless for safety we still skip qnums if they clash with existing qnums
    :param expo_csv: 
    :param database: 
    :param start_qnum: 
    :return: 
    """
    db = QuestionDatabase(location=database, load_expo=False)
    qnums = {q.qnum for q in db.all_questions(unfiltered=True).values()}
    while start_qnum in qnums:
        start_qnum += 1
    curr_qnum = start_qnum

    with open(expo_csv) as f:
        csv_questions = list(csv.DictReader(f))

    questions = []
    for q in csv_questions:
        q['sentences'] = nltk.sent_tokenize(q['text'])
        while curr_qnum in qnums:
            curr_qnum += 1
        qb_question = Question(
            curr_qnum, None, None, None, None, None, q['answer'], 'expo'
        )
        for i, sent in enumerate(q['sentences']):
            qb_question.add_text(i, sent)
        questions.append(qb_question)
        curr_qnum += 1

    return questions


class QuestionDatabase:
    def __init__(self, location=QB_QUESTION_DB, expo_csv=conf['expo_questions'], load_expo=True):
        self._conn = sqlite3.connect(location)
        if os.path.exists(expo_csv) and load_expo:
            self.expo_questions = preprocess_expo_questions(expo_csv)
        else:
            self.expo_questions = []

    def query(self, command: str, arguments) -> Dict[str, Question]:
        questions = {}
        c = self._conn.cursor()
        command = 'select id, page, category, answer, ' + \
            'tournament, naqt, protobowl, fold ' + command
        c.execute(command, arguments)

        for qnum, page, _, answer, tournaments, naqt, protobowl, fold in c:
            questions[qnum] = Question(qnum, answer, None, naqt, protobowl, tournaments, page, fold)

        for q in self.expo_questions:
            questions[q.qnum] = q

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

    @staticmethod
    def normalize_answer(answer):
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

    def questions_with_pages(self) -> Dict[str, List[Question]]:
        page_map = {}

        questions = self.query('from questions where page != ""', ()).values()

        for q in questions:
            page = q.page
            if page not in page_map:
                page_map[page] = []
            page_map[page].append(q)
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

        for q in self.expo_questions:
            answers[q.qnum] = q.page
        return answers


class QuizBowlDataset(AbstractDataset):
    def __init__(self, min_class_examples: int, *,
                 guesser_train=False, buzzer_train=False,
                 qb_question_db: str=QB_QUESTION_DB):
        """
        Initialize a new quiz bowl data set
        :param min_class_examples: The minimum number of training examples to include an answer class.
        """
        super().__init__()
        if not guesser_train and not buzzer_train:
            raise ValueError('Requesting a dataset which produces neither guesser or buzzer training data is invalid')

        if guesser_train and buzzer_train:
            log.warning(
                'Using QuizBowlDataset with guesser and buzzer training data, make sure you know what you are doing!')
        self.db = QuestionDatabase(qb_question_db)
        self.min_class_examples = min_class_examples
        self.guesser_train = guesser_train
        self.buzzer_train = buzzer_train
        self.training_fold = c.GUESSER_TRAIN_FOLD if self.guesser_train else c.BUZZER_TRAIN_FOLD

    def training_data(self) -> TrainingData:
        all_questions = seq(self.db.all_questions().values())
        filtered_questions = all_questions.filter(lambda q: q.fold == self.training_fold)

        if self.min_class_examples > 1:
            filtered_questions = filtered_questions\
                .group_by(lambda q: q.page)\
                .filter(lambda kv: len(kv[1]) >= self.min_class_examples)\
                .flat_map(lambda kv: kv[1])
        filtered_questions = filtered_questions.map(lambda q: q.to_example())
        training_examples = []
        training_answers = []
        for example, answer in filtered_questions:
            training_examples.append(example)
            training_answers.append(answer)

        return training_examples, training_answers

    def questions_by_fold(self, folds=c.ALL_FOLDS) -> Dict[str, List[Question]]:
        all_questions = seq(self.db.all_questions().values())
        train_questions = all_questions\
            .filter(lambda q: q.fold == self.training_fold)\
            .group_by(lambda q: q.page)\
            .filter(lambda kv: len(kv[1]) >= self.min_class_examples)\
            .flat_map(lambda kv: kv[1])\
            .list()

        question_fold_dict = {self.training_fold: train_questions}

        for fold in folds:
            if fold != self.training_fold:
                fold_questions = all_questions.filter(lambda q: q.fold == fold).list()
                question_fold_dict[fold] = fold_questions

        return question_fold_dict

    def questions_in_folds(self, folds: Iterable[str]) -> List[Question]:
        by_fold = self.questions_by_fold(folds=folds)
        questions = []
        for fold in folds:
            questions.extend(by_fold[fold])

        return questions
