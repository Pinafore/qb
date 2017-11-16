import pickle

from qanta.util.constants import DOMAIN_OUTPUT
from qanta.config import conf
from qanta.datasets.abstract import AbstractDataset


class FilteredWikipediaDataset(AbstractDataset):
    def __init__(self, wiki_data_frac=None):
        super().__init__()
        self.wiki_data_frac = wiki_data_frac

    def training_data(self):
        wiki_data_frac = self.wiki_data_frac if self.wiki_data_frac is not None else conf['wiki_data_frac']
        if wiki_data_frac > 0:
            frac = 'frac=' + str(wiki_data_frac)
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
