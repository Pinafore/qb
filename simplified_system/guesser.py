import os
import pickle
from typing import List, Optional, Dict
import progressbar

import elasticsearch
from elasticsearch_dsl import DocType, Text, Keyword, Search, Index
from elasticsearch_dsl.connections import connections

from datasets import QuizBowlDataset
import logging


log = logging.getLogger(__name__)
connections.create_connection(hosts=['localhost'])
INDEX_NAME = 'qb'


class Answer(DocType):
    page = Text(fields={'raw': Keyword()})
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
    def build_large_docs(documents: Dict[str, str], use_qb=True, rebuild_index=False):
        if rebuild_index or bool(int(os.getenv('QB_REBUILD_INDEX', 0))):
            log.info('Deleting index: {}'.format(INDEX_NAME))
            ElasticSearchIndex.delete()

        if ElasticSearchIndex.exists():
            log.info('Index {} exists'.format(INDEX_NAME))
        else:
            log.info('Index {} does not exist'.format(INDEX_NAME))
            Answer.init()
            log.info('Indexing questions as large docs...')
            bar = progressbar.ProgressBar()
            for page in bar(documents):
                if use_qb:
                    qb_content = documents[page]
                else:
                    qb_content = ''

                answer = Answer(
                    page=page,
                    qb_content=qb_content
                )
                answer.save()

    @staticmethod
    def search(text: str, max_n_guesses: int,
               normalize_score_by_length=False,
               qb_boost=1):
        if qb_boost != 1:
            qb_field = 'qb_content^{}'.format(qb_boost)
        else:
            qb_field = 'qb_content'

        s = Search(index='qb')[0:max_n_guesses].query(
            'multi_match', query=text, fields=[qb_field])
        results = s.execute()
        guess_set = set()
        guesses = []
        if normalize_score_by_length:
            query_length = len(text.split())
        else:
            query_length = 1

        for r in results:
            if r.page in guess_set:
                continue
            else:
                guesses.append((r.page, r.meta.score / query_length))
        return guesses

es_index = ElasticSearchIndex()


class ElasticSearchGuesser:
    def __init__(self):
        super().__init__()
        self.use_qb = True
        self.normalize_score_by_length = True
        self.qb_boost = 1

    def qb_dataset(self):
        return QuizBowlDataset(1, guesser_train=True)

    def parameters(self):
        return {
            'use_qb': self.use_qb,
            'normalize_score_by_length': self.normalize_score_by_length,
            'qb_boost': self.qb_boost,
        }

    def train(self, training_data):
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
            rebuild_index=False
        )

    def guess(self, question, max_n_guesses=10):
        return es_index.search(question, max_n_guesses,
                            normalize_score_by_length=self.normalize_score_by_length,
                            qb_boost=self.qb_boost)


if __name__ == '__main__':
    esguesser = ElasticSearchGuesser()
    qb_dataset = QuizBowlDataset(1, guesser_train=True)
    training_data = qb_dataset.training_data()
    esguesser.train(training_data)
    print(training_data[0][0])
    print(training_data[1][0])
    print(esguesser.guess(' '.join(training_data[0][0])))
