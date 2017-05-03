from typing import Dict
import pickle

from elasticsearch_dsl import DocType, Text, Keyword, Search, Index
from elasticsearch_dsl.connections import connections
import elasticsearch
import progressbar

from qanta import logging
from qanta.spark import create_spark_context
from qanta.wikipedia.cached_wikipedia import CachedWikipedia


log = logging.get(__name__)

connections.create_connection(hosts=['localhost'])


class Answer(DocType):
    page = Text(fields={'raw': Keyword()})
    wiki_content = Text()
    qb_content = Text()

    class Meta:
        index = 'qb'


class ElasticSearchIndex:
    @staticmethod
    def build(documents: Dict[str, str], use_all_wikipedia=False, n_cores=4):
        try:
            Index('qb').delete()
        except elasticsearch.exceptions.NotFoundError:
            log.info('Could not delete non-existent index, creating new index...')
        Answer.init()
        cw = CachedWikipedia()
        log.info('Indexing questions and corresponding wikipedia pages...')
        bar = progressbar.ProgressBar()
        for page in bar(documents):
            answer = Answer(page=page, wiki_content=cw[page].content, qb_content=documents[page])
            answer.save()

        if use_all_wikipedia:
            log.info('Indexing all wikipedia pages (particularly those not in training set)...')
            with open('data/external/wiki_pages.pickle', 'rb') as f:
                wiki_pages = pickle.load(f)
            indexed_pages = set(documents.keys())
            pages_to_index = wiki_pages - indexed_pages
            sc = create_spark_context(configs=[('spark.executor.cores', n_cores), ('spark.executor.memory', '4g')])

            def index_page(page):
                answer = Answer(page=page, wiki_content=cw[page].content)
                answer.save()

            sc.parallelize(pages_to_index, 16 * n_cores).foreach(index_page)

    @staticmethod
    def search(text: str, max_n_guesses: int):
        s = Search(index='qb')[0:max_n_guesses].query(
            'multi_match', query=text, fields=['wiki_content', 'qb_content'])
        results = s.execute()
        return [(r.page, r.meta.score) for r in results]
