from abc import ABCMeta, abstractmethod
from typing import Tuple, List, Any, Dict, Optional

Sentence = str
Page = str
Evidence = Dict[str, Any]
TrainingData = Tuple[List[List[Sentence]], List[Page], Optional[List[Evidence]]]


class AbstractDataset(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def training_data(self) -> TrainingData:
        pass
