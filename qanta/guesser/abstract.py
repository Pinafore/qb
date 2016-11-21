from abc import ABCMeta, abstractmethod
from typing import List, Dict, Tuple

from qanta.datasets import TrainingDataset, QuestionText, Answer, Datasets


class AbstractGuesser(metaclass=ABCMeta):
    def __init__(self):
        """
        Abstract class representing a guesser. All abstract methods must be implemented. Class
        construction should be light and not load data since this is reserved for the
        AbstractGuesser.load method.

        self.parallel tells qanta whether or not this guesser should be parallelized.

        self.requested_datasets determines the source of training data given to
        AbstractGuesser.train
        """
        self.parallel = True
        self.requested_datasets = [Datasets.QUIZ_BOWL]

    @abstractmethod
    def train(self, training_data: Dict[str, TrainingDataset]) -> None:
        """
        Given training data, train this guesser so that it can produce guesses.

        The training_data dictionary is keyed by constants from Datasets such as Datasets.QUIZ_BOWL.
        The provided data to this method is based on the requested list of datasets from
        self.requested_datasets.

        The values of these keys is a tuple of two elements which can be seen as (train_x, train_y).
        In this case train_x is a list of question runs. For example, if the answer for a question
        is "Albert Einstein" the runs might be ["This", "This German", "This German physicist", ...]
        train_y is a list of true labels. The questions are strings and the true labels are strings.
        Labels are in canonical form. Questions are not preprocessed in any way. To implement common
        pre-processing refer to the qanta/guesser/preprocessing module.

        :param training_data: training data in the format described above
        :return: This function does not return anything
        """
        pass

    @abstractmethod
    def guess(self,
              questions: List[QuestionText], n_guesses: int) -> List[List[Tuple[Answer, float]]]:
        """
        Given a list of questions as text, return n_guesses number of guesses per question. Guesses
        must be returned in canonical form, are returned with a score in which higher is better, and
        must also be returned in sorted order with the best guess (highest score) at the front of
        the list and worst guesses (lowest score) at the bottom.

        It is guaranteed that before AbstractGuesser.guess is called that either
        AbstractGuesser.train is called or AbstractGuesser.load is called.

        :param questions: Questions to guess on
        :param n_guesses: Number of guesses to produce per question
        :return: List of top guesses per question
        """
        pass

    @abstractmethod
    def score(self, question: str, guesses: List[Answer]) -> List[float]:
        """
        Given a question and a set of guesses, return the score for each guess

        :param question: question to score guesses with
        :param guesses: list of guesses to score
        :return: list of scores corresponding to each guess, in order
        """
        pass

    @staticmethod
    @abstractmethod
    def load(directory: str):
        """
        Given the directory used for saving this guesser, create a new instance of the guesser, and
        load it for guessing or scoring.

        :param directory: training data for guesser
        :return: Instance of AbstractGuesser ready for calling guess/score
        """
        pass

    @abstractmethod
    def save(self, directory: str) -> None:
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """
        Return the display name of this guesser which is used in reporting scripts to identify this
        particular guesser
        :return: display name of this guesser
        """
        pass

