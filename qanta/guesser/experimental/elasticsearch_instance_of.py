import pickle
import os
from typing import List, Optional, Dict
from collections import namedtuple

import elasticsearch
from elasticsearch_dsl.connections import connections
from elasticsearch_dsl import DocType, Text, Keyword, Search, Index
import progressbar
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from qanta.datasets.abstract import QuestionText
from qanta.spark import create_spark_context
from qanta.datasets.quiz_bowl import QuizBowlDataset
from qanta.guesser.abstract import AbstractGuesser
from qanta.config import conf
from qanta import logging
from qanta.wikipedia.cached_wikipedia import CachedWikipedia
from qanta.util.constants import WIKI_INSTANCE_OF_PICKLE
from qanta.wikipedia.wikidata import NO_MATCH


log = logging.get(__name__)

connections.create_connection(hosts=['localhost'])


Claim = namedtuple('Claim', 'item property object title')

GUESSER_PICKLE = 'es_wiki_guesser.pickle'
INDEX_NAME = 'qb_ir_instance_of'


class Answer(DocType):
    page = Text(fields={'raw': Keyword()})
    wiki_content = Text()
    qb_content = Text()
    instance_of = Keyword()

    class Meta:
        index = INDEX_NAME


class ElasticSearchIndex:
    @staticmethod
    def delete():
        try:
            Index(INDEX_NAME).delete()
        except elasticsearch.exceptions.NotFoundError:
            log.info('Could not delete non-existent index')

    @staticmethod
    def exists():
        return Index(INDEX_NAME).exists()

    @staticmethod
    def build(documents: Dict[str, str], instance_of_map, rebuild_index=False):
        if rebuild_index or bool(int(os.getenv('QB_REBUILD_INDEX', 0))):
            log.info('Deleting index: {}'.format(INDEX_NAME))
            ElasticSearchIndex.delete()

        if ElasticSearchIndex.exists():
            log.info('Index {} exists, skipping building index'.format(INDEX_NAME))
        else:
            log.info('Index {} does not exist, building index...'.format(INDEX_NAME))
            Answer.init()
            cw = CachedWikipedia()
            bar = progressbar.ProgressBar()
            for page in bar(documents):
                if page in instance_of_map:
                    instance_of = instance_of_map[page]
                else:
                    instance_of = NO_MATCH
                answer = Answer(
                    page=page,
                    wiki_content=cw[page].content,
                    qb_content=documents[page],
                    instance_of=instance_of
                )
                answer.save()

    def search(self, text: str, predicted_instance_of: str,
               instance_of_probability: float, confidence_threshold: float,
               normalize_score_by_length=True):
        if predicted_instance_of == NO_MATCH:
            apply_filter = False
        elif instance_of_probability > confidence_threshold:
            apply_filter = True
        else:
            apply_filter = False

        if normalize_score_by_length:
            query_length = len(text.split())
        else:
            query_length = 1

        if apply_filter:
            s = Search(index=INDEX_NAME)\
                .filter('term', instance_of=predicted_instance_of)\
                .query(
                    'multi_match',
                    query=text,
                    fields=['wiki_content', 'qb_content']
                )
        else:
            s = Search(index=INDEX_NAME) \
                .query(
                'multi_match',
                query=text,
                fields=['wiki_content', 'qb_content']
            )
        results = s.execute()
        return [(r.page, r.meta.score / query_length) for r in results]


es_index = ElasticSearchIndex()


def format_training_data(instance_of_map: Dict[str, str], questions: List[List[str]], pages: List[str]):
    x_data = []
    y_data = []
    classes = set(instance_of_map.values())
    class_to_i = {label: i for i, label in enumerate(classes, 1)}
    i_to_class = {i: label for label, i in class_to_i.items()}

    for q, p in zip(questions, pages):
        for sent in q:
            x_data.append(sent)
            if p in instance_of_map:
                y_data.append(class_to_i[instance_of_map[p]])
            else:
                y_data.append(class_to_i[NO_MATCH])

    return x_data, y_data, i_to_class, class_to_i


class ElasticSearchWikidataGuesser(AbstractGuesser):
    def __init__(self):
        super().__init__()
        self.class_to_i = None
        self.i_to_class = None
        self.instance_of_model = None
        guesser_conf = conf['guessers']['ESWikidata']
        self.confidence_threshold = guesser_conf['confidence_threshold']
        self.normalize_score_by_length = guesser_conf['normalize_score_by_length']

    def qb_dataset(self):
        return QuizBowlDataset(1, guesser_train=True)

    @classmethod
    def targets(cls):
        return [GUESSER_PICKLE]

    def parameters(self):
        return {
            'confidence_threshold': self.confidence_threshold
        }

    def save(self, directory: str):
        data = {
            'class_to_i': self.class_to_i,
            'i_to_class': self.i_to_class,
            'instance_of_model': self.instance_of_model
        }
        data_pickle_path = os.path.join(directory, GUESSER_PICKLE)
        with open(data_pickle_path, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, directory: str):
        data_pickle_path = os.path.join(directory, GUESSER_PICKLE)
        with open(data_pickle_path, 'rb') as f:
            data = pickle.load(f)
        guesser = ElasticSearchWikidataGuesser()
        guesser.class_to_i = data['class_to_i']
        guesser.i_to_class = data['i_to_class']
        guesser.instance_of_model = data['instance_of_model']
        return guesser

    def train_instance_of(self, instance_of_map, training_data):
        log.info('Creating training data...')
        x_data, y_data, i_to_class, class_to_i = format_training_data(
            instance_of_map, training_data[0], training_data[1]
        )
        self.i_to_class = i_to_class
        self.class_to_i = class_to_i

        log.info('Training instance_of classifier')
        # These parameters have been separately tuned on cross validated scores, they are not random or merely guesses
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(min_df=2, ngram_range=(1, 2))),
            ('classifier', LogisticRegression(C=10))
        ])
        self.instance_of_model = pipeline.fit(x_data, y_data)

    def test_instance_of(self, x_test):
        predictions = self.instance_of_model.predict(x_test)
        probabilities = self.instance_of_model.predict_proba(x_test).max(axis=1)
        class_with_probability = []
        for pred, prob in zip(predictions, probabilities):
            class_with_probability.append((self.i_to_class[pred], prob))
        return class_with_probability

    def train(self, training_data):
        with open(WIKI_INSTANCE_OF_PICKLE, 'rb') as f:
            instance_of_map = pickle.load(f)

        log.info('Building Elastic Search Index...')
        documents = {}
        for sentences, page in zip(training_data[0], training_data[1]):
            paragraph = ' '.join(sentences)
            if page in documents:
                documents[page] += ' ' + paragraph
            else:
                documents[page] = paragraph
        ElasticSearchIndex.build(documents, instance_of_map)

        self.train_instance_of(instance_of_map, training_data)

    def guess(self,
              questions: List[QuestionText],
              max_n_guesses: Optional[int]):
        log.info('Predicting the instance_of attribute for guesses...')
        class_with_probability = self.test_instance_of(questions)

        n_cores = conf['guessers']['ESWikidata']['n_cores']
        sc = create_spark_context(configs=[('spark.executor.cores', n_cores), ('spark.executor.memory', '20g')])

        def ir_search(query_class_and_prob):
            query, class_and_prob = query_class_and_prob
            p_class, prob = class_and_prob
            return es_index.search(
                query, p_class, prob, self.confidence_threshold,
                normalize_score_by_length=self.normalize_score_by_length
            )[:max_n_guesses]

        spark_input = list(zip(questions, class_with_probability))
        log.info('Filtering when classification probability > {}'.format(self.confidence_threshold))

        return sc.parallelize(spark_input, 32 * n_cores).map(ir_search).collect()

    def guess_single(self, question: QuestionText):
        p_class, prob = self.test_instance_of([question])[0]
        guesses = es_index.search(
                question, p_class, prob, self.confidence_threshold,
                normalize_score_by_length=self.normalize_score_by_length)
        return dict(guesses)
