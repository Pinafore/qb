from abc import ABCMeta, abstractmethod

class AbstractDataset(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def training_data(self) -> TrainingDataset:
        pass