from typing import List, Dict

from functional import seq

from qanta.datasets.abstract import AbstractDataset, TrainingData
from qanta.util.environment import QB_QUESTION_DB
from qanta.util.qdb import QuestionDatabase, Question


class QuizBowlDataset(AbstractDataset):
    def __init__(self, min_class_examples: int, qb_question_db: str=QB_QUESTION_DB):
        """
        Initialize a new quiz bowl data set
        :param min_class_examples: The minimum number of training examples to include an answer class.
        """
        super().__init__()
        self.db = QuestionDatabase(qb_question_db)
        self.min_class_examples = min_class_examples

    def training_data(self) -> TrainingData:
        all_questions = seq(self.db.all_questions().values())
        filtered_questions = all_questions\
            .filter(lambda q: q.fold != 'test' and q.fold != 'devtest')\
            .group_by(lambda q: q.page)\
            .filter(lambda kv: len(kv[1]) >= self.min_class_examples)\
            .flat_map(lambda kv: kv[1])\
            .filter(lambda q: q.fold == 'train')\
            .map(lambda q: q.to_example())
        training_examples = []
        training_answers = []
        for example, answer in filtered_questions:
            training_examples.append(example)
            training_answers.append(answer)

        return training_examples, training_answers

    def questions_by_fold(self) -> Dict[str, List[Question]]:
        all_questions = seq(self.db.all_questions().values())
        train_dev_questions = all_questions\
            .filter(lambda q: q.fold == 'train' or q.fold == 'dev')\
            .group_by(lambda q: q.page)\
            .filter(lambda kv: len(kv[1]) >= self.min_class_examples)\
            .flat_map(lambda kv: kv[1])\
            .cache()

        train_questions = train_dev_questions.filter(lambda q: q.fold == 'train').list()
        dev_questions = train_dev_questions.filter(lambda q: q.fold == 'dev').list()

        test_questions = all_questions.filter(lambda q: q.fold == 'test').list()
        devtest_questions = all_questions.filter(lambda q: q.fold == 'devtest').list()

        return {
            'train': train_questions,
            'dev': dev_questions,
            'test': test_questions,
            'devtest': devtest_questions
        }

    def questions_in_folds(self, folds: List[str]) -> List[Question]:
        by_fold = self.questions_by_fold()
        questions = []
        for fold in folds:
            questions += by_fold[fold]

        return questions
