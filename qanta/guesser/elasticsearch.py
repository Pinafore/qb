from typing import List, Optional, Dict
import subprocess
import os
import pickle
import numpy as np

import click
from elasticsearch_dsl import DocType, Text, Keyword, Search, Index
from elasticsearch_dsl.connections import connections
import elasticsearch
import tqdm
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


def create_doctype(index_name, similarity):
    if similarity == 'default':
        wiki_content_field = Text()
        qb_content_field = Text()
    else:
        wiki_content_field = Text(similarity=similarity)
        qb_content_field = Text(similarity=similarity)

    class Answer(DocType):
        page = Text(fields={'raw': Keyword()})
        wiki_content = wiki_content_field
        qb_content = qb_content_field

        class Meta:
            index = index_name

    return Answer


class ElasticSearchIndex:
    def __init__(self, name='qb', similarity='default', bm25_b=None, bm25_k1=None):
        self.name = name
        self.ix = Index(self.name)
        self.answer_doc = create_doctype(self.name, similarity)
        if bm25_b is None:
            bm25_b = .75
        if bm25_k1 is None:
            bm25_k1 = 1.2
        self.bm25_b = bm25_b
        self.bm25_k1 = bm25_k1

    def delete(self):
        try:
            self.ix.delete()
        except elasticsearch.exceptions.NotFoundError:
            log.info('Could not delete non-existent index.')

    def exists(self):
        return self.ix.exists()

    def init(self):
        self.ix.create()
        self.ix.close()
        self.ix.put_settings(body={'similarity': {
            'qb_bm25': {'type': 'BM25', 'b': self.bm25_b, 'k1': self.bm25_k1}}
        })
        self.ix.open()
        self.answer_doc.init(index=self.name)

    def build_large_docs(self, documents: Dict[str, str], use_wiki=True, use_qb=True, rebuild_index=False):
        if rebuild_index or bool(int(os.getenv('QB_REBUILD_INDEX', 0))):
            log.info(f'Deleting index: {self.name}')
            self.delete()

        if self.exists():
            log.info(f'Index {self.name} exists')
        else:
            log.info(f'Index {self.name} does not exist')
            self.init()
            wiki_lookup = Wikipedia()
            log.info('Indexing questions and corresponding wikipedia pages as large docs...')
            for page in tqdm.tqdm(documents):
                if use_wiki and page in wiki_lookup:
                    wiki_content = wiki_lookup[page].text
                else:
                    wiki_content = ''

                if use_qb:
                    qb_content = documents[page]
                else:
                    qb_content = ''

                answer = self.answer_doc(
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
            self.init()
            log.info('Indexing questions and corresponding pages as many docs...')
            if use_qb:
                log.info('Indexing questions...')
                for page, doc in tqdm.tqdm(documents):
                    self.answer_doc(page=page, qb_content=doc).save()

            if use_wiki:
                log.info('Indexing wikipedia...')
                wiki_lookup = Wikipedia()
                for page in tqdm.tqdm(pages):
                    if page in wiki_lookup:
                        content = word_tokenize(wiki_lookup[page].text)
                        for i in range(0, len(content), 200):
                            chunked_content = content[i:i + 200]
                            if len(chunked_content) > 0:
                                self.answer_doc(page=page, wiki_content=' '.join(chunked_content)).save()

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
            'multi_match', query=text, fields=[wiki_field, qb_field]
        )
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
        similarity = guesser_conf['similarity']
        self.similarity_name = similarity['name']
        if self.similarity_name == 'BM25':
            self.similarity_k1 = similarity['k1']
            self.similarity_b = similarity['b']
        else:
            self.similarity_k1 = None
            self.similarity_b = None
        self.index = ElasticSearchIndex(
            name=f'qb_{self.config_num}', similarity=self.similarity_name,
            bm25_b=self.similarity_b, bm25_k1=self.similarity_k1
        )

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
        guesser.similarity_name = params['similarity_name']
        guesser.similarity_b = params['similarity_b']
        guesser.similarity_k1 = params['similarity_k1']

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
                'config_num': self.config_num,
                'similarity_name': self.similarity_name,
                'similarity_k1': self.similarity_k1,
                'similarity_b': self.similarity_b
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

        @app.route('/api/interface_get_highlights', methods=['POST'])
        def get_highlights():
            wiki_field = 'wiki_content'
            qb_field = 'qb_content'
            text = request.form['text']
            s = Search(index='qb')[0:20].query(
                'multi_match', query=text, fields=[wiki_field, qb_field])
            s = s.highlight(wiki_field).highlight(qb_field)
            results = list(s.execute())

            if len(results) == 0:
                highlights = {'wiki': [''],
                              'qb': [''],
                              'guess': ''}
            else:
                guessForEvidence = request.form['guessForEvidence']
                guessForEvidence = guessForEvidence.split("style=\"color:blue\">")[1].split("</a>")[0].lower()

                guess = None
                for index, item in enumerate(results):
                    if item.page.lower().replace("_", " ")[0:25]  == guessForEvidence:
                        guess = results[index]
                        break
                if guess == None:
                    print("expanding search")
                    s = Search(index='qb')[0:80].query(
                        'multi_match', query=text, fields=[wiki_field, qb_field])
                    s = s.highlight(wiki_field).highlight(qb_field)
                    results = list(s.execute()) 
                    for index, item in enumerate(results):
                        if item.page.lower().replace("_", " ")[0:25]  == guessForEvidence:
                            guess = results[index]
                            break
                    if guess == None:
                        highlights = {'wiki': [''],
                                  'qb': [''],
                                  'guess': ''}
                        return jsonify(highlights)
 
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

        @app.route('/api/interface_answer_question', methods=['POST'])
        def answer_question():
            text = request.form['text']
            answer = request.form['answer']
            answer = answer.replace(" ", "_").lower()
            guesses = self.guess([text], 20)[0]

            score_fn = []
            sum_normalize = 0.0
            for (g,s) in guesses:
                exp = np.exp(3*float(s))
                score_fn.append(exp)
                sum_normalize += exp
            for index, (g,s) in enumerate(guesses):
                guesses[index] = (g, score_fn[index] / sum_normalize)

            guess = []
            score = []
            answer_found = False
            num = 0
            for index, (g,s) in enumerate(guesses):
                if index >= 5:
                    break
                guess.append(g)
                score.append(float(s))
            for gue in guess:
                if (gue.lower() == answer.lower()):
                    answer_found = True
                    num = -1
            if (not answer_found):
                for index, (g,s) in enumerate(guesses):
                    if (g.lower() == answer.lower()):
                        guess.append(g)
                        score.append(float(s))
                        num = index + 1
            if (num == 0):
                print("num was 0")
                if (request.form['bell'] == 'true'):
                    return "Num0"
            guess = [g.replace("_"," ") for g in guess]
            return jsonify({'guess': guess, 'score': score, 'num': num})


@click.command()
@click.option('--generate-config/--no-generate-config', default=True, is_flag=True)
@click.option('--config-dir', default='.')
@click.option('--pid-file', default='elasticsearch.pid')
@click.argument('command', type=click.Choice(['start', 'stop', 'configure']))
def elasticsearch_cli(generate_config, config_dir, pid_file, command):
    if generate_config:
        create_es_config(os.path.join(config_dir, 'elasticsearch.yml'))

    if command == 'configure':
        return

    if command == 'start':
        start_elasticsearch(config_dir, pid_file)
    elif command == 'stop':
        stop_elasticsearch(pid_file)
