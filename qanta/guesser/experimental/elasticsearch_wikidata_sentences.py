from typing import List, Optional, Dict, Tuple
import os
import pickle

import elasticsearch
import progressbar
from elasticsearch_dsl import DocType, Text, Keyword, Search, Index
from elasticsearch_dsl.connections import connections

from qanta.datasets.abstract import QuestionText
from qanta.datasets.quiz_bowl import QuizBowlDataset
from qanta.wikipedia.cached_wikipedia import CachedWikipedia
from qanta.guesser.abstract import AbstractGuesser
from qanta.spark import create_spark_context
from qanta.preprocess import format_guess
from qanta.config import conf
from qanta import logging


log = logging.get(__name__)


connections.create_connection(hosts=['localhost'])


class Answer(DocType):
    page = Text(fields={'raw': Keyword()})
    wiki_content = Text()
    qb_content = Text()
    wikidata_sentences = Text()

    class Meta:
        index = 'qb'


class ElasticSearchIndex:
    @staticmethod
    def build(documents: Dict[str, Tuple[str, str]], wiki_sentences_enabled=True, wiki_page_enabled=True):
        try:
            Index('qb').delete()
        except elasticsearch.exceptions.NotFoundError:
            log.info('Could not delete non-existent index, creating new index...')
        Answer.init()
        cw = CachedWikipedia()
        log.info('Indexing questions and corresponding wikipedia pages...')
        bar = progressbar.ProgressBar()
        for page in bar(documents):
            qb_content, wikidata_sentences = documents[page]
            if not wiki_sentences_enabled:
                wikidata_sentences = ''
            if wiki_page_enabled:
                wiki_page = cw[page].content
            else:
                wiki_page = ''
            answer = Answer(
                page=page, wiki_content=wiki_page,
                qb_content=qb_content, wikidata_sentences=wikidata_sentences
            )
            answer.save()

    @staticmethod
    def search(text: str, max_n_guesses: int):
        s = Search(index='qb')[0:max_n_guesses].query(
            'multi_match', query=text, fields=['wiki_content', 'qb_content', 'wikidata_sentences'])
        results = s.execute()
        return [(r.page, r.meta.score) for r in results]


ES_PARAMS = 'es_params.pickle'
es_index = ElasticSearchIndex()


class ElasticSearchWikiSentencesGuesser(AbstractGuesser):
    def __init__(self):
        super().__init__()
        guesser_conf = conf['guessers']['ElasticSearchWikiSentences']
        self.min_appearances = guesser_conf['min_appearances']
        self.n_cores = guesser_conf['n_cores']
        self.wiki_page_enabled = guesser_conf['wiki_page_enabled']
        self.wiki_sentences_enabled = guesser_conf['wiki_sentences_enabled']

    def qb_dataset(self):
        return QuizBowlDataset(self.min_appearances)

    def parameters(self):
        return {
            'min_appearances': self.min_appearances,
            'n_cores': self.n_cores,
            'wiki_page_enabled': self.wiki_page_enabled,
            'wiki_sentences_enabled': self.wiki_sentences_enabled
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

        with open('/ssd-c/qanta/page_sentences.pickle', 'rb') as f:
            page_sentences = pickle.load(f)

        n = 0
        for page in documents:
            if page in page_sentences:
                n += 1
                w_sentences = ' '.join(page_sentences[page])
                w_sentences = ' '.join(set(w_sentences.lower().split()))
            else:
                w_sentences = ''
            documents[page] = (documents[page], w_sentences)

        log.info('Added wiki sentences to {} of {} pages'.format(n, len(documents)))

        ElasticSearchIndex.build(
            documents,
            wiki_sentences_enabled=self.wiki_sentences_enabled,
            wiki_page_enabled=self.wiki_page_enabled
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
        with open(os.path.join(directory, ES_PARAMS), 'rb') as f:
            params = pickle.load(f)
        guesser = ElasticSearchWikiSentencesGuesser()
        guesser.min_appearances = params['min_appearances']
        guesser.n_cores = params['n_cores']
        guesser.wiki_page_enabled = params['wiki_page_enabled']
        guesser.wiki_sentences_enabled = params['wiki_sentences_enabled']
        return guesser

    def save(self, directory: str):
        with open(os.path.join(directory, ES_PARAMS), 'wb') as f:
            pickle.dump({
                'min_appearances': self.min_appearances,
                'n_cores': self.n_cores,
                'wiki_page_enabled': self.wiki_page_enabled,
                'wiki_sentences_enabled': self.wiki_sentences_enabled
            }, f)
