from typing import Dict, List, Tuple
from qanta.guesser.abstract import AbstractGuesser

import tensorflow as tf


class DANGuesser(AbstractGuesser):
    def __init__(self):
        pass

    def train(self,
              training_data: Dict[str, Tuple[List[List[str]], List[str]]]) -> None:
        pass

    def load(self, directory: str) -> None:
        pass

    def guess(self, questions: List[str], n_guesses: int) -> List[
        List[Tuple[str, float]]]:
        pass

    @property
    def display_name(self) -> str:
        return 'DAN'

    def save(self, directory: str) -> None:
        pass

    def score(self, question: str, guesses: List[str]) -> List[float]:
        pass

