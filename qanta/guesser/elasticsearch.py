from typing import List, Optional, Tuple, Dict
import os
import pickle

from elasticsearch_dsl import DocType, Text, Keyword, Search, Index
from elasticsearch_dsl.connections import connections
import elasticsearch
import progressbar
from nltk.tokenize import word_tokenize

from qanta.wikipedia.cached_wikipedia import CachedWikipedia
from qanta.datasets.abstract import QuestionText
from qanta.datasets.quiz_bowl import QuizBowlDataset
from qanta.guesser.abstract import AbstractGuesser
from qanta.spark import create_spark_context
from qanta.config import conf
from qanta import logging


log = logging.get(__name__)
connections.create_connection(hosts=['localhost'])
INDEX_NAME = 'qb'


class Answer(DocType):
    page = Text(fields={'raw': Keyword()})
    wiki_content = Text()
    qb_content = Text()

    class Meta:
        index = INDEX_NAME


class ElasticSearchIndex:
    @staticmethod
    def delete():
        try:
            Index(INDEX_NAME).delete()
        except elasticsearch.exceptions.NotFoundError:
            log.info('Could not delete non-existent index, creating new index...')

    @staticmethod
    def exists():
        return Index(INDEX_NAME).exists()

    @staticmethod
    def build_large_docs(documents: Dict[str, str], use_wiki=True, use_qb=True, rebuild_index=False):
        if rebuild_index:
            ElasticSearchIndex.delete()
        Answer.init()
        cw = CachedWikipedia()
        log.info('Indexing questions and corresponding wikipedia pages as large docs...')
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
    def build_many_docs(pages, documents, use_wiki=True, use_qb=True, rebuild_index=False):
        if rebuild_index:
            ElasticSearchIndex.delete()

        Answer.init()
        log.info('Indexing questions and corresponding pages as many docs...')
        if use_qb:
            log.info('Indexing questions...')
            bar = progressbar.ProgressBar()
            for page, doc in bar(documents):
                Answer(page=page, qb_content=doc).save()

        if use_wiki:
            log.info('Indexing wikipedia')
            cw = CachedWikipedia()
            bar = progressbar.ProgressBar()
            for page in bar(pages):
                content = word_tokenize(cw[page].content)
                for i in range(0, len(content), 200):
                    chunked_content = content[i:i + 200]
                    if len(chunked_content) > 0:
                        Answer(page=page, wiki_content=' '.join(chunked_content)).save()

    @staticmethod
    def search(text: str, max_n_guesses: int):
        s = Search(index='qb')[0:max_n_guesses].query(
            'multi_match', query=text, fields=['wiki_content', 'qb_content'])
        results = s.execute()
        guess_set = set()
        guesses = []
        for r in results:
            if r.page in guess_set:
                continue
            else:
                guesses.append((r.page, r.meta.score))
        return guesses

ES_PARAMS = 'es_params.pickle'
es_index = ElasticSearchIndex()


class ElasticSearchGuesser(AbstractGuesser):
    def __init__(self):
        super().__init__()
        guesser_conf = conf['guessers']['ElasticSearch']
        self.n_cores = guesser_conf['n_cores']
        self.use_wiki = guesser_conf['use_wiki']
        self.use_qb = guesser_conf['use_qb']
        self.many_docs = guesser_conf['many_docs']

    def qb_dataset(self):
        return QuizBowlDataset(1, guesser_train=True)

    def parameters(self):
        return {
            'n_cores': self.n_cores,
            'use_wiki': self.use_wiki,
            'use_qb': self.use_qb,
            'many_docs': self.many_docs
        }

    def train(self, training_data):
        if self.many_docs:
            pages = set(training_data[1])
            documents = []
            for sentences, page in zip(training_data[0], training_data[1]):
                paragraph = ' '.join(sentences)
                documents.append((page, paragraph))
            ElasticSearchIndex.build_many_docs(pages, documents, use_qb=self.use_qb, use_wiki=self.use_wiki)
        else:
            documents = {}
            for sentences, page in zip(training_data[0], training_data[1]):
                paragraph = ' '.join(sentences)
                if page in documents:
                    documents[page] += ' ' + paragraph
                else:
                    documents[page] = paragraph

            ElasticSearchIndex.build_large_docs(
                documents,
                use_qb=self.use_qb,
                use_wiki=self.use_wiki
            )

    def guess(self, questions: List[QuestionText], max_n_guesses: Optional[int]):
        sc = create_spark_context(configs=[('spark.executor.cores', self.n_cores), ('spark.executor.memory', '20g')])

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
        guesser.n_cores = params['n_cores']
        guesser.use_wiki = params['use_wiki']
        guesser.use_qb = params['use_qb']
        guesser.many_docs = params['many_docs']
        return guesser

    def save(self, directory: str):
        with open(os.path.join(directory, ES_PARAMS), 'wb') as f:
            pickle.dump({
                'n_cores': self.n_cores,
                'use_wiki': self.use_wiki,
                'use_qb': self.use_qb,
                'many_docs': self.many_docs
            }, f)
