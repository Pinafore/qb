from typing import List, Dict, Iterable, Optional, Tuple, NamedTuple
import json

from qanta import qlogging
from qanta.datasets.abstract import AbstractDataset, TrainingData
from qanta.util.constants import (
    QANTA_MAPPED_DATASET_PATH,
    GUESSER_TRAIN_FOLD, GUESSER_DEV_FOLD, BUZZER_TRAIN_FOLD, BUZZER_DEV_FOLD,
    SYSTEM_DEV_FOLD, SYSTEM_TEST_FOLD,
    TRAIN_FOLDS, DEV_FOLDS, ALL_FOLDS
)


log = qlogging.get(__name__)


class Question(NamedTuple):
    qanta_id: int
    text: str
    first_sentence: str
    tokenizations: List[Tuple[int, int]]
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
    def __init__(self, *, guesser_train=False, buzzer_train=False) -> None:
        """
        Initialize a new quiz bowl data set
        """
        super().__init__()
        if not guesser_train and not buzzer_train:
            raise ValueError('Requesting a dataset which produces neither guesser or buzzer training data is invalid')

        if guesser_train and buzzer_train:
            log.warning(
                'Using QuizBowlDataset with guesser and buzzer training data, make sure you know what you are doing!'
            )

        self.db = QantaDatabase()
        self.guesser_train = guesser_train
        self.buzzer_train = buzzer_train


    def training_data(self) -> TrainingData:
        training_examples = []
        training_pages = []
        questions = []  # type: List[Question]
        if self.guesser_train:
            questions.extend(self.db.guess_train_questions)
        if self.buzzer_train:
            questions.extend(self.db.buzz_train_questions)

        for q in questions:
            training_examples.append(q.text)
            training_pages.append(q.page)

        return training_examples, training_pages, None

    def questions_by_fold(self, folds=ALL_FOLDS) -> Dict[str, List[Question]]:
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
