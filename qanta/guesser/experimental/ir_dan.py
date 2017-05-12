from typing import List, Optional, Dict
import os
import pickle

from elasticsearch_dsl import DocType, Text, Keyword, Search, Index
from elasticsearch_dsl.connections import connections
import elasticsearch
import progressbar

from qanta.datasets.abstract import QuestionText
from qanta.datasets.quiz_bowl import QuizBowlDataset
from qanta.guesser.abstract import AbstractGuesser
from qanta.preprocess import format_guess
from qanta.spark import create_spark_context
from qanta.wikipedia.cached_wikipedia import CachedWikipedia

from qanta.config import conf
from qanta import logging


log = logging.get(__name__)
connections.create_connection(hosts=['localhost'])
index_name = 'ir_dan_guesser'

class Answer(DocType):
    page = Text(fields={'raw': Keyword()})
    wiki_content = Text()
    qb_content = Text()

    class Meta:
        index = index_name


class ElasticSearchIndex:
    @staticmethod
    def build(documents: Dict[str, str], use_wiki=True, use_qb=True, n_cores=4):
        try:
            Index('qb').delete()
        except elasticsearch.exceptions.NotFoundError:
            log.info('Could not delete non-existent index, creating new index...')
        Answer.init()
        cw = CachedWikipedia()
        log.info('Indexing questions and corresponding wikipedia pages...')
        bar = progressbar.ProgressBar()
        for page in bar(documents):
            if use_wiki:
                wiki_content = cw[page].content
            else:
                wiki_content = ''

            if use_qb:
                qb_content = documents[page]
            else:
                qb_content = ''

            answer = Answer(page=page, wiki_content=wiki_content, qb_content=qb_content)
            answer.save()

    @staticmethod
    def search(text: str, max_n_guesses: int):
        s = Search(index=index_name)[0:max_n_guesses].query(
            'multi_match', query=text, fields=['wiki_content', 'qb_content'])
        results = s.execute()
        return [(r.page, r.meta.score) for r in results]

IRDAN_PARAMS = 'irdan_params.pickle'
es_index = ElasticSearchIndex()


class IrDanGuesser(AbstractGuesser):
    def __init__(self):
        super().__init__()
        guesser_conf = conf['guessers']['ElasticSearch']
        self.min_appearances = guesser_conf['min_appearances']
        self.n_cores = guesser_conf['n_cores']
        self.use_wiki = guesser_conf['use_wiki']
        self.use_qb = guesser_conf['use_qb']

    def qb_dataset(self):
        return QuizBowlDataset(self.min_appearances)

    def parameters(self):
        return {
            'min_appearances': self.min_appearances,
            'n_cores': self.n_cores,
            'use_wiki': self.use_wiki,
            'use_qb': self.use_qb
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

        ElasticSearchIndex.build(
            documents,
            n_cores=self.n_cores,
            use_qb=self.use_qb,
            use_wiki=self.use_wiki
        )

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
        with open(os.path.join(directory, IRDAN_PARAMS), 'rb') as f:
            params = pickle.load(f)
        guesser = IrDanGuesser()
        guesser.min_appearances = params['min_appearances']
        guesser.n_cores = params['n_cores']
        return guesser

    def save(self, directory: str):
        with open(os.path.join(directory, IRDAN_PARAMS), 'wb') as f:
            pickle.dump({
                'min_appearances': self.min_appearances,
                'n_cores': self.n_cores
            }, f)