import abc
from typing import List

from qanta.util.constants import N_GUESSES


class AbstractFeatureExtractor(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def name(self):
        pass

    @abc.abstractmethod
    def score_guesses(self, guesses: List[str], text: str) -> List[str]:
        """
        Given a list of guesses to score on the given text, return vowpal wabbit features in
        their own namespace
        :param guesses: guesses to score
        :param text: question text to score against
        :return: list of vowpal wabbit features
        """
        pass
