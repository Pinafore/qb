import pickle
from collections import defaultdict

from qanta import logging
from qanta.util.constants import DOMAIN_OUTPUT
from qanta.config import conf
from qanta.datasets.abstract import AbstractDataset, TrainingData


log = logging.get(__name__)


def compute_ans_distribution(training_data: TrainingData):
    questions, answers, *_ = training_data
    counts = defaultdict(int)
    for q, a in zip(questions, answers):
        counts[a] += len(q)
    return counts


def compute_threshold_distribution(ans_dist, threshold):
    """
    Given an answer distribution and a threshold, return a dictionary whose keys are answers and values
    are how many questions need to be added to the training data in order for at least threshold questions to exist
    per question
    :param ans_dist:
    :param threshold:
    :return:
    """
    add_dist = {}
    for ans, n in ans_dist.items():
        if n < threshold:
            add_dist[ans] = threshold - n

    return add_dist


class FilteredWikipediaDataset(AbstractDataset):
    def __init__(self, add_dist=None):
        super().__init__()
        self.add_dist = dict(add_dist)
        self.wiki_data_frac = conf['wiki_data_frac']

    def training_data(self):
        if self.wiki_data_frac > 0:
            n_added = 0
            frac = 'frac=' + str(self.wiki_data_frac)
            with open(DOMAIN_OUTPUT.format(frac), 'rb') as f:
                data = pickle.load(f)
            x = []
            y = []
            for text, page in data:
                if self.add_dist is None:
                    n_added += 1
                    x.append([text])
                    y.append(page)
                else:
                    if page in self.add_dist and self.add_dist[page] > 0:
                        n_added += 1
                        x.append([text])
                        y.append(page)
                        self.add_dist[page] -= 1
            log.info(f'Added {n_added} sentences from wikipedia')

            return x, y, [None for _ in x]
        else:
            return [], [], []
