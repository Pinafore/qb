import pickle

from qanta.util.constants import DOMAIN_OUTPUT
from qanta.config import conf
from qanta.datasets.abstract import AbstractDataset


class FilteredWikipediaDataset(AbstractDataset):
    def __init__(self):
        super().__init__()

    def training_data(self):
        if conf['wiki_data_frac'] > 0:
            frac = 'frac=' + str(conf['wiki_data_frac'])
            with open(DOMAIN_OUTPUT.format(frac), 'rb') as f:
                data = pickle.load(f)
            x = []
            y = []
            for text, page in data:
                x.append([text])
                y.append(page)

            return x, y, [None for _ in x]
        else:
            return [], [], []
