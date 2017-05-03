from abc import ABCMeta, abstractmethod
from typing import Tuple, List, Dict, Optional

QuestionText = str
Answer = str
TrainingData = Tuple[List[List[QuestionText]], List[Answer], Optional[List[Dict]]]


class AbstractDataset(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def training_data(self) -> TrainingData:
        pass
