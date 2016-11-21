from abc import ABCMeta, abstractmethod
from typing import Tuple, List

QuestionText = str
Answer = str

TrainingDataset = Tuple[List[List[QuestionText]], List[Answer]]


class Datasets:
    QUIZ_BOWL = 'QUIZ_BOWL:MIN_ANSWERS=5'
    WIKI = 'WIKI'



