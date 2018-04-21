from typing import List, Dict, Iterable, Optional, Any, Set, NamedTuple
import json
from collections import defaultdict, Counter

from qanta import qlogging
from qanta.datasets.abstract import AbstractDataset, TrainingData
from qanta.util.constants import (
    QANTA_MAPPED_DATASET_PATH,
    GUESSER_TRAIN_FOLD, GUESSER_DEV_FOLD, BUZZER_TRAIN_FOLD, BUZZER_DEV_FOLD,
    SYSTEM_DEV_FOLD, SYSTEM_TEST_FOLD,
    TRAIN_FOLDS, DEV_FOLDS
)
from qanta.config import conf


log = qlogging.get(__name__)


class Question(NamedTuple):
    qanta_id: int
    text: str
    first_sentence: str
    first_end_char: int
    answer: str
    page: Optional[str]
    fold: str
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

    @classmethod
    def from_dict(cls, dict_question):
        return cls(**dict_question)

    def to_dict(self):
        return self._asdict()


class QantaDatabase:
    def __init__(self, dataset_path=QANTA_MAPPED_DATASET_PATH):
        with open(dataset_path) as f:
            self.dataset = json.load(f)

        self.version = self.dataset['version']
        self.raw_questions = self.dataset['questions']
        self.all_questions = [Question(**q) for q in self.raw_questions]
        self.mapped_questions = [q for q in self.all_questions if q.page is not None]

        self.train_questions = [q for q in self.mapped_questions if q.fold in TRAIN_FOLDS]
        self.guess_train_questions = [q for q in self.train_questions if q.fold == GUESSER_TRAIN_FOLD]
        self.buzz_train_questions = [q for q in self.train_questions if q.fold == BUZZER_TRAIN_FOLD]

        self.dev_questions = [q for q in self.mapped_questions if q.fold in DEV_FOLDS]
        self.guess_dev_questions = [q for q in self.dev_questions if q.fold == GUESSER_DEV_FOLD]
        self.buzz_dev_questions = [q for q in self.dev_questions if q.fold == BUZZER_DEV_FOLD]
        self.system_dev_questions = [q for q in self.dev_questions if q.fold == SYSTEM_DEV_FOLD]

        self.test_questions = [q for q in self.mapped_questions if q.fold == SYSTEM_TEST_FOLD]


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
