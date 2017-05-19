import abc
from typing import List
import warnings


warnings.warn('old features extractors are deprecated and need to be rewritten', DeprecationWarning)


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
