from typing import Dict, List, Tuple
from collections import defaultdict
import pickle
import os
import random

import numpy as np
import luigi

from qanta.guesser.abstract import AbstractGuesser
from qanta.datasets.abstract import AbstractDataset
from qanta.datasets.quiz_bowl import QuizBowlDataset


class RandomGuesser(AbstractGuesser):
    """
    A simple reference guesser that guesses randomly based on the observed answer distribution
    """

    ANSWER_LOOKUP_FILE = 'answer_lookup.pickle'
    ANSWER_PDF_FILE = 'answer_pdf.pickle'
    ANSWER_SET_FILE = 'answer_set.pickle'

    def __init__(self):
        super().__init__()
        self.answer_lookup = {}
        self.answer_pdf = []
        self.answer_set = set()

    @property
    def requested_datasets(self) -> Dict[str, AbstractDataset]:
        return {
            'qb': QuizBowlDataset(5)
        }

    @property
    def display_name(self) -> str:
        return 'Random'

    def train(self,
              training_data: Dict[str, Tuple[List[List[str]], List[str]]]) -> None:
        answer_counts = defaultdict(int)
        for dataset in training_data:
            examples, labels = training_data[dataset]
            for answer in labels:
                answer_counts[answer] += 1

        for i, answer in enumerate(answer_counts):
            self.answer_lookup[answer] = i
            self.answer_pdf.append(answer_counts[answer])

        self.answer_pdf = np.array(self.answer_pdf) / np.sum(self.answer_pdf)
        self.answer_set = set(self.answer_lookup)

    def guess(self, questions: List[str], n_guesses: int) -> List[List[Tuple[str, float]]]:
        guesses = []
        for q in questions:
            random.seed(hash(q))
            q_guesses = []
            for guess in random.sample(self.answer_set, n_guesses):
                q_guesses.append((guess, self.answer_pdf[self.answer_lookup[guess]]))
            guesses.append(q_guesses)

        return guesses

    def score(self, question: str, guesses: List[str]) -> List[float]:
        scores = []
        for guess in guesses:
            if guess in self.answer_lookup:
                scores.append(self.answer_pdf[self.answer_lookup[guess]])
            else:
                scores.append(0)

        return scores

    @staticmethod
    def files(directory: str) -> None:
        output_files = [
            RandomGuesser.ANSWER_LOOKUP_FILE,
            RandomGuesser.ANSWER_PDF_FILE,
            RandomGuesser.ANSWER_SET_FILE
        ]
        return [os.path.join(directory, file) for file in output_files]

    def save(self, directory: str) -> None:
        with open(os.path.join(directory, RandomGuesser.ANSWER_LOOKUP_FILE), 'wb') as f:
            pickle.dump(self.answer_lookup, f)

        with open(os.path.join(directory, RandomGuesser.ANSWER_PDF_FILE), 'wb') as f:
            pickle.dump(self.answer_pdf, f)

        with open(os.path.join(directory, RandomGuesser.ANSWER_SET_FILE), 'wb') as f:
            pickle.dump(self.answer_set, f)

    @staticmethod
    def load(directory: str):
        guesser = RandomGuesser()

        with open(os.path.join(directory, RandomGuesser.ANSWER_LOOKUP_FILE), 'rb') as f:
            guesser.answer_lookup = pickle.load(f)

        with open(os.path.join(directory, RandomGuesser.ANSWER_PDF_FILE), 'rb') as f:
            guesser.answer_pdf = pickle.load(f)

        with open(os.path.join(directory, RandomGuesser.ANSWER_SET_FILE), 'rb') as f:
            guesser.answer_set = pickle.load(f)

        return guesser
