from typing import List, Optional

from qanta.datasets.abstract import QuestionText
from qanta.datasets.quiz_bowl import QuizBowlDataset
from qanta.guesser.abstract import AbstractGuesser
from qanta.spark import create_spark_context
from qanta.preprocess import format_guess
from qanta.search.elasticsearch import ElasticSearchIndex
from qanta.config import conf
from qanta import logging


log = logging.get(__name__)


es_index = ElasticSearchIndex()


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
        n_cores = conf['guessers']['ElasticSearch']['n_cores']
        sc = create_spark_context(configs=[('spark.executor.cores', n_cores), ('spark.executor.memory', '4g')])

        def es_search(query):
            return es_index.search(query, max_n_guesses)

        return sc.parallelize(questions, 16 * n_cores).map(es_search).collect()

    @classmethod
    def targets(cls):
        return []

    @classmethod
    def load(cls, directory: str):
        return ElasticSearchGuesser()

    def save(self, directory: str):
        pass
