from abc import ABCMeta, abstractmethod
from typing import Tuple, List, Any, Dict, Optional

QuestionText = str
Page = str
Evidence = Dict[str, Any]
TrainingData = Tuple[List[List[QuestionText]], List[Page], Optional[List[Evidence]]]


class AbstractDataset(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def training_data(self) -> TrainingData:
        pass
