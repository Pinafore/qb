from typing import List, Optional
import os
import pickle

from qanta.datasets.abstract import QuestionText
from qanta.datasets.quiz_bowl import QuizBowlDataset
from qanta.guesser.abstract import AbstractGuesser
from qanta.spark import create_spark_context
from qanta.preprocess import format_guess
from qanta.search.elasticsearch import ElasticSearchIndex
from qanta.config import conf
from qanta import logging


log = logging.get(__name__)

ES_PARAMS = 'es_params.pickle'
es_index = ElasticSearchIndex()


class ElasticSearchGuesser(AbstractGuesser):
    def __init__(self):
        super().__init__()
        guesser_conf = conf['guessers']['ElasticSearch']
        self.use_all_wikipedia = guesser_conf['use_all_wikipedia']
        self.min_appearances = guesser_conf['min_appearances']
        self.n_cores = guesser_conf['n_cores']

    def qb_dataset(self):
        return QuizBowlDataset(self.min_appearances)

    def parameters(self):
        return {
            'use_all_wikipedia': self.use_all_wikipedia,
            'min_appearances': self.min_appearances,
            'n_cores': self.n_cores
        }

    def train(self, training_data):
        documents = {}
        for sentences, ans in zip(training_data[0], training_data[1]):
            page = format_guess(ans)
            paragraph = ' '.join(sentences)
            if page in documents:
                documents[page] += ' ' + paragraph
            else:
                documents[page] = paragraph

        ElasticSearchIndex.build(documents, use_all_wikipedia=self.use_all_wikipedia, n_cores=self.n_cores)

    def guess(self, questions: List[QuestionText], max_n_guesses: Optional[int]):
        sc = create_spark_context(configs=[('spark.executor.cores', self.n_cores), ('spark.executor.memory', '4g')])

        def es_search(query):
            return es_index.search(query, max_n_guesses)

        return sc.parallelize(questions, 16 * self.n_cores).map(es_search).collect()

    @classmethod
    def targets(cls):
        return []

    @classmethod
    def load(cls, directory: str):
        with open(os.path.join(directory, ES_PARAMS), 'rb') as f:
            params = pickle.load(f)
        guesser = ElasticSearchGuesser()
        guesser.use_all_wikipedia = params['use_all_wikipedia']
        guesser.min_appearances = params['min_appearances']
        guesser.n_cores = params['n_cores']
        return guesser

    def save(self, directory: str):
        with open(os.path.join(directory, ES_PARAMS), 'wb') as f:
            pickle.dump({
                'use_all_wikipedia': self.use_all_wikipedia,
                'min_appearances': self.min_appearances,
                'n_cores': self.n_cores
            }, f)
