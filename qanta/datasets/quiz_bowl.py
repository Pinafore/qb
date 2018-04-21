from typing import List, Dict, Iterable, Tuple, Optional, Any, Set, NamedTuple
import json
import csv
import os
import sqlite3
from collections import defaultdict, Counter

import nltk

from qanta import qlogging
from qanta.datasets.abstract import AbstractDataset, TrainingData, QuestionText, Answer
from qanta.util.environment import QB_QUESTION_DB
from qanta.util import constants as c
from qanta.util.io import file_backed_cache_decorator, safe_path
from qanta.config import conf


log = qlogging.get(__name__)


class Question(NamedTuple):
    qanta_id: int
    text: str
    first_sentence: str
    first_end_char: int
    answer: str
    page: Optional[str]
    category: Optional[str]
    subcategory: Optional[str]
    tournament: str
    difficulty: str
    year: int
    proto_id: Optional[int]
    qdb_id: Optional[int]
    dataset: str

    def to_json(self):
        return json.dumps(self._asdict())

    @classmethod
    def from_json(cls, json_text):
        return cls(**json.loads(json_text))


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
        self.location = location
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

        for qnum, page, category, answer, tournaments, naqt, protobowl, fold in c:
            questions[qnum] = Question(qnum, answer, category, naqt, protobowl, tournaments, page, fold)

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
        page_map = defaultdict(list) # type: Dict[str, List[Question]]
        questions = self.query('from questions where page != ""', ()).values()

        for q in questions:
            page = q.page
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
    def __init__(self, *, guesser_train=False, buzzer_train=False,
                 qb_question_db: str=QB_QUESTION_DB) -> None:
        """
        Initialize a new quiz bowl data set
        """
        super().__init__()
        if not guesser_train and not buzzer_train:
            raise ValueError('Requesting a dataset which produces neither guesser or buzzer training data is invalid')

        if guesser_train and buzzer_train:
            log.warning(
                'Using QuizBowlDataset with guesser and buzzer training data, make sure you know what you are doing!')
        self.db = QuestionDatabase(qb_question_db)
        self.guesser_train = guesser_train
        self.buzzer_train = buzzer_train
        self.training_folds = set() # type: Set[str]
        if self.guesser_train:
            self.training_folds.add(c.GUESSER_TRAIN_FOLD)
        if self.buzzer_train:
            self.training_folds.add(c.BUZZER_TRAIN_FOLD)


    def training_data(self) -> TrainingData:
        from functional import seq
        all_questions = seq(self.db.all_questions().values())
        all_evidence = None  # type: Optional[Dict[str, Any]]

        filtered_questions = all_questions\
            .filter(lambda q: q.fold in self.training_folds)\
            .map(lambda q: q.to_example(all_evidence=all_evidence))
        training_examples = []
        training_answers = []
        training_evidence = []
        for example, answer, evidence in filtered_questions:
            training_examples.append(example)
            training_answers.append(answer)
            training_evidence.append(evidence)

        return training_examples, training_answers, training_evidence

    def questions_by_fold(self, folds=c.ALL_FOLDS) -> Dict[str, List[Question]]:
        from functional import seq
        all_questions = seq(self.db.all_questions().values())
        train_questions = all_questions\
            .filter(lambda q: q.fold in self.training_folds)\
            .group_by(lambda q: q.page)\
            .flat_map(lambda kv: kv[1])\
            .list()

        if len(self.training_folds) == 1:
            fold = next(iter(self.training_folds))
            question_fold_dict = {fold: train_questions}
        else:
            question_fold_dict = {'guessertrain': train_questions}

        for fold in folds:
            if fold not in self.training_folds:
                fold_questions = all_questions.filter(lambda q: q.fold == fold).list()
                question_fold_dict[fold] = fold_questions

        return question_fold_dict

    def questions_in_folds(self, folds: Iterable[str]) -> List[Question]:
        by_fold = self.questions_by_fold(folds=folds)
        questions = []
        for fold in folds:
            questions.extend(by_fold[fold])

        return questions
