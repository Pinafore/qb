from typing import List, Optional, Dict
import os
import pickle

from elasticsearch_dsl import DocType, Text, Keyword, Search, Index
from elasticsearch_dsl.connections import connections
import elasticsearch
import progressbar
from nltk.tokenize import word_tokenize

from qanta.wikipedia.cached_wikipedia import Wikipedia
from qanta.datasets.abstract import QuestionText
from qanta.datasets.quiz_bowl import QuizBowlDataset
from qanta.guesser.abstract import AbstractGuesser
from qanta.spark import create_spark_context
from qanta.config import conf
from qanta import qlogging


log = qlogging.get(__name__)
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
        if rebuild_index or bool(int(os.getenv('QB_REBUILD_INDEX', 0))):
            log.info('Deleting index: {}'.format(INDEX_NAME))
            ElasticSearchIndex.delete()

        if ElasticSearchIndex.exists():
            log.info('Index {} exists'.format(INDEX_NAME))
        else:
            log.info('Index {} does not exist'.format(INDEX_NAME))
            Answer.init()
            wiki_lookup = Wikipedia()
            log.info('Indexing questions and corresponding wikipedia pages as large docs...')
            bar = progressbar.ProgressBar()
            for page in bar(documents):
                if use_wiki and page in wiki_lookup:
                    wiki_content = wiki_lookup[page].text
                else:
                    wiki_content = ''

                if use_qb:
                    qb_content = documents[page]
                else:
                    qb_content = ''

                answer = Answer(
                    page=page,
                    wiki_content=wiki_content, qb_content=qb_content
                )
                answer.save()

    @staticmethod
    def build_many_docs(pages, documents, use_wiki=True, use_qb=True, rebuild_index=False):
        if rebuild_index or bool(int(os.getenv('QB_REBUILD_INDEX', 0))):
            log.info('Deleting index: {}'.format(INDEX_NAME))
            ElasticSearchIndex.delete()

        if ElasticSearchIndex.exists():
            log.info('Index {} exists'.format(INDEX_NAME))
        else:
            log.info('Index {} does not exist'.format(INDEX_NAME))
            Answer.init()
            log.info('Indexing questions and corresponding pages as many docs...')
            if use_qb:
                log.info('Indexing questions...')
                bar = progressbar.ProgressBar()
                for page, doc in bar(documents):
                    Answer(page=page, qb_content=doc).save()

            if use_wiki:
                log.info('Indexing wikipedia...')
                wiki_lookup = Wikipedia()
                bar = progressbar.ProgressBar()
                for page in bar(pages):
                    if page in wiki_lookup:
                        content = word_tokenize(wiki_lookup[page].text)
                        for i in range(0, len(content), 200):
                            chunked_content = content[i:i + 200]
                            if len(chunked_content) > 0:
                                Answer(page=page, wiki_content=' '.join(chunked_content)).save()

    @staticmethod
    def search(text: str, max_n_guesses: int,
               normalize_score_by_length=False,
               wiki_boost=1, qb_boost=1):
        if wiki_boost != 1:
            wiki_field = 'wiki_content^{}'.format(wiki_boost)
        else:
            wiki_field = 'wiki_content'

        if qb_boost != 1:
            qb_field = 'qb_content^{}'.format(qb_boost)
        else:
            qb_field = 'qb_content'

        s = Search(index='qb')[0:max_n_guesses].query(
            'multi_match', query=text, fields=[wiki_field, qb_field])
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
        self.normalize_score_by_length = guesser_conf['normalize_score_by_length']
        self.qb_boost = guesser_conf['qb_boost']
        self.wiki_boost = guesser_conf['wiki_boost']

    def qb_dataset(self):
        return QuizBowlDataset(guesser_train=True)

    def parameters(self):
        return {
            'n_cores': self.n_cores,
            'use_wiki': self.use_wiki,
            'use_qb': self.use_qb,
            'many_docs': self.many_docs,
            'normalize_score_by_length': self.normalize_score_by_length,
            'qb_boost': self.qb_boost,
            'wiki_boost': self.wiki_boost
        }

    def train(self, training_data):
        if self.many_docs:
            pages = set(training_data[1])
            documents = []
            for sentences, page in zip(training_data[0], training_data[1]):
                paragraph = ' '.join(sentences)
                documents.append((page, paragraph))
            ElasticSearchIndex.build_many_docs(
                pages, documents,
                use_qb=self.use_qb, use_wiki=self.use_wiki
            )
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
        def es_search(query):
                return es_index.search(query, max_n_guesses,
                                       normalize_score_by_length=self.normalize_score_by_length,
                                       wiki_boost=self.wiki_boost, qb_boost=self.qb_boost)

        if len(questions) > 1:
            sc = create_spark_context(configs=[('spark.executor.cores', self.n_cores), ('spark.executor.memory', '20g')])
            return sc.parallelize(questions, 16 * self.n_cores).map(es_search).collect()
        elif len(questions) == 1:
            return [es_search(questions[0])]
        else:
            return []

    @classmethod
    def targets(cls):
        return []

    @classmethod
    def load(cls, directory: str):
        with open(os.path.join(directory, ES_PARAMS), 'rb') as f:
            params = pickle.load(f)
        guesser = ElasticSearchGuesser()
        guesser.use_wiki = params['use_wiki']
        guesser.use_qb = params['use_qb']
        guesser.many_docs = params['many_docs']
        guesser.normalize_score_by_length = params['normalize_score_by_length']
        return guesser

    def save(self, directory: str):
        with open(os.path.join(directory, ES_PARAMS), 'wb') as f:
            pickle.dump({
                'use_wiki': self.use_wiki,
                'use_qb': self.use_qb,
                'many_docs': self.many_docs,
                'normalize_score_by_length': self.normalize_score_by_length
            }, f)

    def web_api(self, host='0.0.0.0', port=5000, debug=False):
        from flask import Flask, jsonify, request

        app = Flask(__name__)

        @app.route('/api/answer_question', methods=['POST'])
        def answer_question():
            text = request.form['text']
            guess, score = self.guess([text], 1)[0][0]
            return jsonify({'guess': guess, 'score': float(score)})

        @app.route('/api/get_highlights', methods=['POST'])
        def get_highlights():
            wiki_field = 'wiki_content'
            qb_field = 'qb_content'
            text = request.form['text']
            s = Search(index='qb')[0:10].query(
                'multi_match', query=text, fields=[wiki_field, qb_field])
            s = s.highlight(wiki_field).highlight(qb_field)
            results = list(s.execute())

            if len(results) == 0:
                highlights = {'wiki': [''],
                              'qb': [''],
                              'guess': ''}
            else:
                guess = results[0] # take the best answer
                _highlights = guess.meta.highlight 
                try:
                    wiki_content = list(_highlights.wiki_content)
                except AttributeError:
                    wiki_content = ['']

                try:
                    qb_content = list(_highlights.qb_content)
                except AttributeError:
                    qb_content = ['']

                highlights = {'wiki': wiki_content,
                              'qb': qb_content,
                              'guess': guess.page}
            return jsonify(highlights)

        app.run(host=host, port=port, debug=debug)
