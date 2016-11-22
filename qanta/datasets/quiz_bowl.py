from functional import seq

from qanta.datasets.abstract import AbstractDataset, TrainingData
from qanta.util.environment import QB_QUESTION_DB
from qanta.util.qdb import QuestionDatabase


class QuizBowlDataset(AbstractDataset):
    def __init__(self, min_class_examples: int):
        """
        Initialize a new quiz bowl data set
        :param min_class_examples: The minimum number of training examples to include an answer class.
        """
        super().__init__()
        self.db = QuestionDatabase(QB_QUESTION_DB)
        self.min_class_examples = min_class_examples

    def training_data(self) -> TrainingData:
        all_questions = seq(self.db.all_questions().values())
        filtered_questions = all_questions\
            .filter(lambda q: q.fold != 'test' and q.fold != 'devtest')\
            .group_by(lambda q: q.page)\
            .filter(lambda kv: len(kv[1]) >= self.min_class_examples)\
            .flat_map(lambda kv: kv[1])\
            .map(lambda q: q.to_example())
        training_examples = []
        training_answers = []
        for example, answer in filtered_questions:
            training_examples.append(example)
            training_answers.append(answer)

        return training_examples, training_answers
