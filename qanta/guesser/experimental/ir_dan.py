from typing import List, Optional, Dict
import os
import pickle

from elasticsearch_dsl import DocType, Text, Keyword, Search, Index
from elasticsearch_dsl.connections import connections
import elasticsearch
import progressbar
import numpy as np

from qanta.datasets.abstract import QuestionText
from qanta.datasets.quiz_bowl import QuizBowlDataset
from qanta.guesser.abstract import AbstractGuesser
from qanta.spark import create_spark_context
from qanta.wikipedia.cached_wikipedia import CachedWikipedia
from qanta.config import conf
from qanta.preprocess import preprocess_dataset, tokenize_question
from qanta.guesser import nn
from qanta import logging


log = logging.get(__name__)
connections.create_connection(hosts=['localhost'])
index_name = 'ir_dan_guesser'

IR_DAN_WE_TMP = '/tmp/qanta/deep/ir_dan_we.pickle'
IR_DAN_WE = 'ir_dan_we.pickle'

load_embeddings = nn.create_load_embeddings_function(IR_DAN_WE_TMP, IR_DAN_WE, log)


class Answer(DocType):
    page = Text(fields={'raw': Keyword()})
    wiki_content = Text()

    class Meta:
        index = index_name


class ElasticSearchIndex:
    @staticmethod
    def delete():
        try:
            Index(index_name).delete()
        except elasticsearch.exceptions.NotFoundError:
            log.info('Count not delete non-existent index: {}'.format(index_name))

    @staticmethod
    def exists():
        return Index(index_name).exists()

    @staticmethod
    def build(pages, rebuild_index=False):
        if rebuild_index or os.getenv('QB_REBUILD_INDEX', False):
            ElasticSearchIndex.delete()

        if ElasticSearchIndex.exists():
            log.info('Found index, skipping building index')
        else:
            Answer.init()
            cw = CachedWikipedia()
            log.info('Indexing questions and corresponding wikipedia pages...')
            bar = progressbar.ProgressBar()
            for page in bar(pages):
                wiki_content = cw[page].content
                answer = Answer(page=page, wiki_content=wiki_content)
                answer.save()

    @staticmethod
    def search(text: str, max_n_guesses: int):
        s = Search(index=index_name)[0:max_n_guesses].query(
            'multi_match', query=text, fields=['wiki_content'])
        results = s.execute()
        return [(r.page, r.meta.score) for r in results]

IRDAN_PARAMS = 'irdan_params.pickle'
es_index = ElasticSearchIndex()


def preprocess_wikipedia(all_top_guesses, cached_wikipedia, vocab):
    rows = []
    for q_guesses in all_top_guesses:
        q_rows = []
        for page, _ in q_guesses:
            content = tokenize_question(cached_wikipedia[page])[:150]
            q_rows.append(content)
            for t in content:
                vocab.append(t)
        rows.append(q_rows)
    return rows


class IrDanGuesser(AbstractGuesser):
    def __init__(self):
        super().__init__()
        guesser_conf = conf['guessers']['ElasticSearch']
        self.min_appearances = guesser_conf['min_appearances']
        self.n_cores = guesser_conf['n_cores']
        self.use_wiki = guesser_conf['use_wiki']
        self.use_qb = guesser_conf['use_qb']
        self.embeddings = None
        self.embedding_lookup = None
        self.n_classes = None
        self.max_len = None

    def qb_dataset(self):
        return QuizBowlDataset(self.min_appearances, guesser_train=True)

    def parameters(self):
        return {
            'min_appearances': self.min_appearances,
            'n_cores': self.n_cores,
            'use_wiki': self.use_wiki,
            'use_qb': self.use_qb
        }

    def build_model(self):
        pass

    def train(self, training_data):
        log.info('Building elastic search index...')
        ElasticSearchIndex.build(set(training_data[1]))

        def fetch_top_guesses(query):
            return ElasticSearchIndex.search(query, 10)

        sentences = []
        for q in training_data[0]:
            for s in q:
                sentences.append(s)

        sc = create_spark_context('QB: IR NN Re-ranker')
        all_top_guesses = sc.parallelize(sentences).map(fetch_top_guesses).collect()
        sc.stop()

        log.info('Processing training data...')
        x_train, y_train, x_test, y_test, vocab, class_to_i, i_to_class = preprocess_dataset(
            training_data
        )

        cw = CachedWikipedia()
        all_top_pages = preprocess_wikipedia(all_top_guesses, cw, vocab)

        log.info('Creating embeddings...')
        embeddings, embedding_lookup = load_embeddings(vocab=vocab, expand_glove=True, mask_zero=True)
        self.embeddings = embeddings
        self.embedding_lookup = embedding_lookup

        x_train = [nn.convert_text_to_embeddings_indices(q, embedding_lookup) for q in x_train]
        x_test = [nn.convert_text_to_embeddings_indices(q, embedding_lookup) for q in x_test]
        self.n_classes = nn.compute_n_classes(training_data[1])
        self.max_len = nn.compute_max_len(training_data)
        x_train = np.array(nn.tf_format(x_train, self.max_len, 0))
        x_test = np.array(nn.tf_format(x_test, self.max_len, 0))

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