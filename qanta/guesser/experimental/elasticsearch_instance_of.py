import pydevd
pydevd.settrace('localhost', port=9999, stdoutToServer=True, stderrToServer=True)

import pickle
import json
import os
from typing import List, Optional, Dict
from collections import namedtuple, Counter
from qanta.spark import create_spark_context

import elasticsearch
from elasticsearch_dsl.connections import connections
from elasticsearch_dsl import DocType, Text, Keyword, Search, Index, Boolean
import progressbar

from qanta.datasets.abstract import QuestionText
from qanta.datasets.quiz_bowl import QuizBowlDataset
from qanta.guesser.abstract import AbstractGuesser
from qanta.preprocess import format_guess
from qanta.config import conf
from qanta.util.io import safe_open
from qanta import logging
from qanta.wikipedia.cached_wikipedia import CachedWikipedia


log = logging.get(__name__)

connections.create_connection(hosts=['localhost'])

WIKIDATA_CLAIMS = 'data/external/wikidata-claims_instance-of.jsonl'

Claim = namedtuple('Claim', 'item property object title')


def parse_instance_of_claims():
    ds = QuizBowlDataset(1)
    training_data = ds.training_data()
    answer_set = {format_guess(ans) for ans in training_data[1]}
    claims = []
    with open(WIKIDATA_CLAIMS) as f:
        for l in f:
            c = json.loads(l)
            if c['property'] == 'instance of' and c['object'] != 'Wikimedia disambiguation page':
                claims.append(Claim(c['item'], c['property'], c['object'], c['title']))

    instance_of_types = Counter()
    for c in claims:
        if type(c.object) == str:
            if c.title is not None and format_guess(c.title) in answer_set:
                instance_of_types[c.object] += 1
            elif format_guess(c.item) in answer_set:
                instance_of_types[c.object] += 1
    return claims


class Answer(DocType):
    page = Text(fields={'raw': Keyword()})
    wiki_content = Text()
    qb_content = Text()
    is_human = Boolean()

    class Meta:
        index = 'qb'


class ElasticSearchIndex:
    @staticmethod
    def build(documents: Dict[str, str], is_human_map):
        try:
            Index('qb').delete()
        except elasticsearch.exceptions.NotFoundError:
            log.info('Could not delete non-existent index, creating new index...')
        Answer.init()
        cw = CachedWikipedia()
        bar = progressbar.ProgressBar()
        for page in bar(documents):
            if page in is_human_map:
                is_human = is_human_map[page]
            else:
                is_human = False
            answer = Answer(
                page=page,
                wiki_content=cw[page].content,
                qb_content=documents[page],
                is_human=is_human
            )
            answer.save()

    def search(self, text: str, is_human_probability: float):
        if is_human_probability > .8:
            is_human = True
            apply_filter = True
        elif is_human_probability < .2:
            is_human = False
            apply_filter = True
        else:
            is_human = None
            apply_filter = False
        if apply_filter:
            s = Search(index='qb')\
                .filter('term', is_human=is_human)\
                .query(
                    'multi_match',
                    query=text,
                    fields=['wiki_content', 'qb_content']
                )
        else:
            s = Search(index='qb') \
                .query(
                'multi_match',
                query=text,
                fields=['wiki_content', 'qb_content']
            )
        results = s.execute()
        return [(r.page, r.meta.score) for r in results]


es_index = ElasticSearchIndex()


def format_human_data(is_human_map, questions: List[List[str]], pages: List[str]):
    x_data = []
    y_data = []

    for q, p in zip(questions, pages):
        full_text = ' '.join(q)
        x_data.append(full_text)

        p = format_guess(p)
        if p in is_human_map:
            y_data.append(int(is_human_map[p]))
        else:
            y_data.append(0)

    return x_data, y_data



class ElasticSearchWikidataGuesser(AbstractGuesser):
    def __init__(self, is_human_model=None):
        super().__init__()
        self.is_human_model = is_human_model

    def qb_dataset(self):
        return QuizBowlDataset(conf['guessers']['ElasticSearch']['min_appearances'])

    def train(self, training_data):
        answers = {format_guess(a) for a in training_data[1]}
        log.info('Loading instance of data from wikidata...')
        instance_of_map = create_instance_of_map(answers)
        is_human_map = create_is_human_map(instance_of_map)
        log.info('Creating training data...')
        x_data, y_data = format_human_data(is_human_map, training_data[0], training_data[1])
        log.info('Training is_human model...')
        self.is_human_model = create_human_model(x_data, y_data)

        log.info('Building Elastic Search Index...')
        documents = {}
        for sentences, ans in zip(training_data[0], training_data[1]):
            page = format_guess(ans)
            paragraph = ' '.join(sentences)
            if page in documents:
                documents[page] += ' ' + paragraph
            else:
                documents[page] = paragraph
        ElasticSearchIndex.build(documents, is_human_map)

    def guess(self,
              questions: List[QuestionText],
              max_n_guesses: Optional[int]):
        n_cores = conf['guessers']['ElasticSearch']['n_cores']
        sc = create_spark_context(configs=[('spark.executor.cores', n_cores), ('spark.executor.memory', '40g')])
        b_is_human_model = sc.broadcast(self.is_human_model)

        def ir_search(query):
            is_human_model = b_is_human_model.value
            is_human_probability = is_human_model.predict_proba([query])[0][1]
            return es_index.search(query, is_human_probability)[:max_n_guesses]

        return sc.parallelize(questions, 4 * n_cores).map(ir_search).collect()

    @classmethod
    def targets(cls):
        return [IS_HUMAN_MODEL_PICKLE]

    @classmethod
    def load(cls, directory: str):
        with open(os.path.join(directory, IS_HUMAN_MODEL_PICKLE), 'rb') as f:
            is_human_model = pickle.load(f)['is_human_model']
        return ElasticSearchWikidataGuesser(is_human_model=is_human_model)

    def save(self, directory: str):
        with safe_open(os.path.join(directory, IS_HUMAN_MODEL_PICKLE), 'wb') as f:
            pickle.dump({'is_human_model': self.is_human_model}, f)
