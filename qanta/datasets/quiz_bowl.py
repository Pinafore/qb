from qanta.datasets import AbstractDataset, TrainingDataset


class QuizBowlDataset(AbstractDataset):
    def __init__(self):
        super().__init__()

    def training_data(self) -> TrainingDataset:
        pass

