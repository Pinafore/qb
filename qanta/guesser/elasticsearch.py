from multiprocessing import Pool
from itertools import repeat
from typing import List, Optional
from qanta.datasets.abstract import QuestionText
from qanta.datasets.quiz_bowl import QuizBowlDataset
from qanta.guesser.abstract import AbstractGuesser
from qanta.preprocess import format_guess
from qanta.search.elasticsearch import ElasticSearchIndex
from qanta.config import conf
from qanta import logging


log = logging.get(__name__)


es_index = ElasticSearchIndex()


def es_search(max_n_guesses, query):
    return es_index.search(query)[:max_n_guesses]


class ElasticSearchGuesser(AbstractGuesser):
    def qb_dataset(self):
        return QuizBowlDataset(conf['guessers']['ElasticSearch']['min_appearances'])

    def train(self, training_data):
        documents = {}
        for sentences, ans in zip(training_data[0], training_data[1]):
            page = format_guess(ans)
            paragraph = ' '.join(sentences)
            if page in documents:
                documents[page] += ' ' + paragraph
            else:
                documents[page] = paragraph
        ElasticSearchIndex.build(documents)

    def guess(self,
              questions: List[QuestionText],
              max_n_guesses: Optional[int]):
        predictions = []
        pool = Pool(processes=conf['guessers']['ElasticSearch']['n_cores'])
        for guesses in pool.starmap(es_search,
                                    zip(repeat(max_n_guesses), questions), chunksize=1000):
            predictions.append(guesses)
        return predictions

    @classmethod
    def targets(cls):
        return []

    @classmethod
    def load(cls, directory: str):
        return ElasticSearchGuesser()

    def save(self, directory: str):
        pass
