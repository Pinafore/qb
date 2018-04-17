from typing import List, Optional, Dict
import subprocess
import os
import pickle

from elasticsearch_dsl import DocType, Text, Keyword, Search, Index
from elasticsearch_dsl.connections import connections
import elasticsearch
import progressbar
from nltk.tokenize import word_tokenize
from jinja2 import Environment, PackageLoader

from qanta.wikipedia.cached_wikipedia import Wikipedia
from qanta.datasets.abstract import QuestionText
from qanta.guesser.abstract import AbstractGuesser
from qanta.spark import create_spark_context
from qanta.config import conf
from qanta.util.io import get_tmp_dir, safe_path
from qanta import qlogging


log = qlogging.get(__name__)
ES_PARAMS = 'es_params.pickle'
connections.create_connection(hosts=['localhost'])


def create_es_config(output_path, host='localhost', port=9200, tmp_dir=None):
    if tmp_dir is None:
        tmp_dir = get_tmp_dir()
    data_dir = safe_path(os.path.join(tmp_dir, 'elasticsearch/data/'))
    log_dir = safe_path(os.path.join(tmp_dir, 'elasticsearch/log/'))
    env = Environment(loader=PackageLoader('qanta', 'templates'))
    template = env.get_template('elasticsearch.yml')
    config_content = template.render({
        'host': host,
        'port': port,
        'log_dir': log_dir,
        'data_dir': data_dir
    })
    with open(output_path, 'w') as f:
        f.write(config_content)


def start_elasticsearch(config_dir, pid_file):
    subprocess.run(
        ['elasticsearch', '-d', '-p', pid_file, f'-Epath.conf={config_dir}']
    )


def stop_elasticsearch(pid_file):
    with open(pid_file) as f:
        pid = int(f.read())
    subprocess.run(['kill', str(pid)])


class Answer(DocType):
    page = Text(fields={'raw': Keyword()})
    wiki_content = Text()
    qb_content = Text()


class ElasticSearchIndex:
    def __init__(self, name='qb'):
        self.name = name

    def delete(self):
        try:
            Index(self.name).delete()
        except elasticsearch.exceptions.NotFoundError:
            log.info('Could not delete non-existent index.')

    def exists(self):
        return Index(self.name).exists()

    def build_large_docs(self, documents: Dict[str, str], use_wiki=True, use_qb=True, rebuild_index=False):
        if rebuild_index or bool(int(os.getenv('QB_REBUILD_INDEX', 0))):
            log.info(f'Deleting index: {self.name}')
            self.delete()

        if self.exists():
            log.info(f'Index {self.name} exists')
        else:
            log.info(f'Index {self.name} does not exist')
            Answer.init(index=self.name)
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
                answer.save(index=self.name)

    def build_many_docs(self, pages, documents, use_wiki=True, use_qb=True, rebuild_index=False):
        if rebuild_index or bool(int(os.getenv('QB_REBUILD_INDEX', 0))):
            log.info(f'Deleting index: {self.name}')
            self.delete()

        if self.exists():
            log.info(f'Index {self.name} exists')
        else:
            log.info(f'Index {self.name} does not exist')
            Answer.init(index=self.name)
            log.info('Indexing questions and corresponding pages as many docs...')
            if use_qb:
                log.info('Indexing questions...')
                bar = progressbar.ProgressBar()
                for page, doc in bar(documents):
                    Answer(page=page, qb_content=doc).save(index=self.name)

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
                                Answer(page=page, wiki_content=' '.join(chunked_content)).save(index=self.name)

    def search(self, text: str, max_n_guesses: int,
               normalize_score_by_length=False,
               wiki_boost=1, qb_boost=1):
        if not self.exists():
            raise ValueError('The index does not exist, you must create it before searching')

        if wiki_boost != 1:
            wiki_field = 'wiki_content^{}'.format(wiki_boost)
        else:
            wiki_field = 'wiki_content'

        if qb_boost != 1:
            qb_field = 'qb_content^{}'.format(qb_boost)
        else:
            qb_field = 'qb_content'

        s = Search(index=self.name)[0:max_n_guesses].query(
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


class ElasticSearchGuesser(AbstractGuesser):
    def __init__(self, config_num):
        super().__init__(config_num)
        guesser_conf = conf['guessers']['qanta.guesser.elasticsearch.ElasticSearchGuesser'][self.config_num]
        self.n_cores = guesser_conf['n_cores']
        self.use_wiki = guesser_conf['use_wiki']
        self.use_qb = guesser_conf['use_qb']
        self.many_docs = guesser_conf['many_docs']
        self.normalize_score_by_length = guesser_conf['normalize_score_by_length']
        self.qb_boost = guesser_conf['qb_boost']
        self.wiki_boost = guesser_conf['wiki_boost']
        self.index = ElasticSearchIndex(name=f'qb_{self.config_num}')

    def parameters(self):
        return conf['guessers']['qanta.guesser.elasticsearch.ElasticSearchGuesser'][self.config_num]

    def train(self, training_data):
        if self.many_docs:
            pages = set(training_data[1])
            documents = []
            for sentences, page in zip(training_data[0], training_data[1]):
                paragraph = ' '.join(sentences)
                documents.append((page, paragraph))
            self.index.build_many_docs(
                pages, documents,
                use_qb=self.use_qb, use_wiki=self.use_wiki, rebuild_index=True
            )
        else:
            documents = {}
            for sentences, page in zip(training_data[0], training_data[1]):
                paragraph = ' '.join(sentences)
                if page in documents:
                    documents[page] += ' ' + paragraph
                else:
                    documents[page] = paragraph

            self.index.build_large_docs(
                documents,
                use_qb=self.use_qb,
                use_wiki=self.use_wiki,
                rebuild_index=True
            )


    def guess(self, questions: List[QuestionText], max_n_guesses: Optional[int]):
        def es_search(query):
            return self.index.search(
                query, max_n_guesses,
                normalize_score_by_length=self.normalize_score_by_length,
                wiki_boost=self.wiki_boost, qb_boost=self.qb_boost
            )

        if len(questions) > 1:
            sc = create_spark_context(configs=[('spark.executor.cores', self.n_cores), ('spark.executor.memory', '20g')])
            return sc.parallelize(questions, 16 * self.n_cores).map(es_search).collect()
        elif len(questions) == 1:
            return [es_search(questions[0])]
        else:
            return []

    @classmethod
    def targets(cls):
        return [ES_PARAMS]

    @classmethod
    def load(cls, directory: str):
        with open(os.path.join(directory, ES_PARAMS), 'rb') as f:
            params = pickle.load(f)
        guesser = ElasticSearchGuesser(params['config_num'])
        guesser.n_cores = params['n_cores']
        guesser.use_wiki = params['use_wiki']
        guesser.use_qb = params['use_qb']
        guesser.many_docs = params['many_docs']
        guesser.normalize_score_by_length = params['normalize_score_by_length']
        guesser.qb_boost = params['qb_boost']
        guesser.wiki_boost = params['wiki_boost']
        return guesser

    def save(self, directory: str):
        with open(os.path.join(directory, ES_PARAMS), 'wb') as f:
            pickle.dump({
                'n_cores': self.n_cores,
                'use_wiki': self.use_wiki,
                'use_qb': self.use_qb,
                'many_docs': self.many_docs,
                'normalize_score_by_length': self.normalize_score_by_length,
                'qb_boost': self.qb_boost,
                'wiki_boost': self.wiki_boost,
                'config_num': self.config_num
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
