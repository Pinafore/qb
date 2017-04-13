import pickle

from qanta.util.constants import DOMAIN_OUTPUT
from qanta.datasets.abstract import AbstractDataset

class FilteredWikipediaDataset(AbstractDataset):
    def __init__(self):
        super().__init__()

    def training_data(self):
        with open(DOMAIN_OUTPUT, 'rb') as f:
            data = pickle.load(f)
        x = []
        y = []
        for text, page in data:
            x.append([text])
            y.append(page)

        return x, y, None
