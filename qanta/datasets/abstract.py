from abc import ABCMeta, abstractmethod
from typing import Tuple, List

QuestionText = str
Answer = str
TrainingData = Tuple[List[List[QuestionText]], List[Answer]]


class AbstractDataset(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def training_data(self) -> TrainingData:
        pass
