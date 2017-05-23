import pickle
import random
import os
from typing import List, Optional, Dict
from collections import namedtuple
import re

import elasticsearch
from elasticsearch_dsl.connections import connections
from elasticsearch_dsl import DocType, Text, Keyword, Search, Index, Boolean
import progressbar

from qanta.datasets.abstract import QuestionText
from qanta.spark import create_spark_context
from qanta.datasets.quiz_bowl import QuizBowlDataset
from qanta.guesser.abstract import AbstractGuesser
from qanta.config import conf
from qanta.util.io import shell
from qanta import logging
from qanta.wikipedia.cached_wikipedia import CachedWikipedia
from qanta.util.constants import WIKI_INSTANCE_OF_PICKLE


log = logging.get(__name__)

connections.create_connection(hosts=['localhost'])


Claim = namedtuple('Claim', 'item property object title')

VW_CLASSIFIER_MODEL = 'vw_classifier.model'
VW_CLASSIFIER_PICKLE = 'vw_classifier.pickle'


class Answer(DocType):
    page = Text(fields={'raw': Keyword()})
    wiki_content = Text()
    qb_content = Text()
    instance_of = Boolean()

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


def vw_normalize_string(text):
    return re.sub('\s', ' ', text.lower().replace(':', '').replace('|', ''))


def format_training_data(instance_of_map: Dict[str, str], questions: List[List[str]], pages: List[str]):
    x_data = []
    y_data = []
    classes = set(instance_of_map.values())
    class_to_i = {label: i for i, label in enumerate(classes, 1)}
    i_to_class = {i: label for label, i in class_to_i.items()}

    for q, p in zip(questions, pages):
        for sent in q:
            x_data.append(vw_normalize_string(sent))
            if p in instance_of_map:
                y_data.append(class_to_i[instance_of_map[p]])
            else:
                y_data.append(class_to_i['NO MATCH!'])

    return x_data, y_data, i_to_class, class_to_i


class ElasticSearchWikidataGuesser(AbstractGuesser):
    def __init__(self):
        super().__init__()
        self.class_to_i = None
        self.i_to_class = None
        self.max_class = None
        guesser_conf = conf['guessers']['ESWikidata']
        self.multiclass_one_against_all = guesser_conf['multiclass_one_against_all']
        self.multiclass_online_trees = guesser_conf['multiclass_online_trees']
        self.l1 = guesser_conf['l1']
        self.l2 = guesser_conf['l2']
        self.passes = guesser_conf['passes']
        self.learning_rate = guesser_conf['learning_rate']
        self.decay_learning_rate = guesser_conf['decay_learning_rate']
        self.bits = guesser_conf['bits']
        if not (self.multiclass_one_against_all != self.multiclass_online_trees):
            raise ValueError('The options multiclass_one_against_all and multiclass_online_trees are XOR')

    def qb_dataset(self):
        return QuizBowlDataset(1, guesser_train=True)

    @staticmethod
    def targets(cls):
        return [VW_CLASSIFIER_MODEL, VW_CLASSIFIER_PICKLE]

    def parameters(self):
        return {
            'multiclass_one_against_all': self.multiclass_one_against_all,
            'multiclass_online_trees': self.multiclass_online_trees,
            'l1': self.l1,
            'l2': self.l2,
            'passes': self.passes,
            'learning_rate': self.learning_rate,
            'decay_learning_rate': self.decay_learning_rate,
            'bits': self.bits
        }

    def save(self, directory: str):
        model_path = os.path.join(directory, VW_CLASSIFIER_MODEL)
        shell('cp /tmp/{} {}'.format(VW_CLASSIFIER_MODEL, model_path))
        data = {
            'class_to_i': self.class_to_i,
            'i_to_class': self.i_to_class,
            'max_class': self.max_class,
            'multiclass_one_against_all': self.multiclass_one_against_all,
            'multiclass_online_trees': self.multiclass_online_trees,
            'l1': self.l1,
            'l2': self.l2,
            'passes': self.passes,
            'learning_rate': self.learning_rate,
            'decay_learning_rate': self.decay_learning_rate,
            'bits': self.bits
        }
        data_pickle_path = os.path.join(directory, VW_CLASSIFIER_PICKLE)
        with open(data_pickle_path, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, directory: str):
        model_path = os.path.join(directory, VW_CLASSIFIER_MODEL)
        shell('cp {} /tmp/{}'.format(model_path, VW_CLASSIFIER_MODEL))
        data_pickle_path = os.path.join(directory, VW_CLASSIFIER_PICKLE)
        with open(data_pickle_path, 'rb') as f:
            data = pickle.load(f)
        guesser = ElasticSearchWikidataGuesser()
        guesser.class_to_i = data['class_to_i']
        guesser.i_to_class = data['i_to_class']
        guesser.max_class = data['max_class']
        guesser.multiclass_one_against_all = data['multiclass_one_against_all']
        guesser.multiclass_online_trees = data['multiclass_online_trees']
        guesser.l1 = data['l1']
        guesser.l2 = data['l2']
        guesser.passes = data['passes']
        guesser.learning_rate = data['learning_rate']
        guesser.decay_learning_rate = data['decay_learning_rate']
        guesser.bits = data['bits']
        return guesser

    def vw_train(self, instance_of_map, training_data):
        log.info('Creating training data...')
        x_data, y_data, i_to_class, class_to_i = format_training_data(
            instance_of_map, training_data[0], training_data[1]
        )
        self.i_to_class = i_to_class
        self.class_to_i = class_to_i
        self.max_class = len(self.class_to_i)

        with open('/tmp/vw_train.txt', 'w') as f:
            zipped = list(zip(x_data, y_data))
            random.shuffle(zipped)
            for x, y in zipped:
                f.write('{label} |words {sentence}\n'.format(label=y, sentence=x))

        if self.multiclass_one_against_all:
            multiclass_flag = '--oaa'
        elif self.multiclass_online_trees:
            multiclass_flag = '--log_multi'
        else:
            raise ValueError('The options multiclass_one_against_all and multiclass_online_trees are XOR')

        log.info('Training VW Model...')
        shell('vw -k {multiclass_flag} {max_class} -d /tmp/vw_train.txt -f /tmp/{model} --loss_function logistic '
              '--ngram 1 --ngram 2 --skips 1 -c --passes {passes} -b {bits} '
              '--l1 {l1} --l2 {l2} -l {learning_rate} --decay_learning_rate {decay_learning_rate}'.format(
                    max_class=self.max_class, model=VW_CLASSIFIER_MODEL,
                    multiclass_flag=multiclass_flag, bits=self.bits,
                    l1=self.l1, l2=self.l2, passes=self.passes,
                    learning_rate=self.learning_rate, decay_learning_rate=self.decay_learning_rate
                ))

    def vw_test(self, x_data):
        with open('/tmp/vw_test.txt', 'w') as f:
            for x in x_data:
                f.write('1 |words {text}\n'.format(text=vw_normalize_string(x)))
        shell('vw -t -i /tmp/{model} -p /tmp/predictions.txt -d /tmp/vw_test.txt')
        predictions = []
        with open('/tmp/predictions.txt') as f:
            for line in f:
                label = int(line)
                predictions.append(label)
        return predictions

    def train(self, training_data):
        with open(WIKI_INSTANCE_OF_PICKLE, 'rb') as f:
            instance_of_map = pickle.load(f)

        self.vw_train(instance_of_map, training_data)
        log.info('Building Elastic Search Index...')
        documents = {}
        for sentences, page in zip(training_data[0], training_data[1]):
            paragraph = ' '.join(sentences)
            if page in documents:
                documents[page] += ' ' + paragraph
            else:
                documents[page] = paragraph
        ElasticSearchIndex.build(documents, instance_of_map)

    def guess(self,
              questions: List[QuestionText],
              max_n_guesses: Optional[int]):
        n_cores = conf['guessers']['ElasticSearch']['n_cores']
        sc = create_spark_context(configs=[('spark.executor.cores', n_cores), ('spark.executor.memory', '40g')])
        b_instance_of_model = sc.broadcast(self.instance_of_model)

        def ir_search(query):
            instance_of_model = b_instance_of_model.value
            is_human_probability = instance_of_model.predict_proba([query])[0][1]
            return es_index.search(query, is_human_probability)[:max_n_guesses]

        return sc.parallelize(questions, 4 * n_cores).map(ir_search).collect()

    @classmethod
    def targets(cls):
        return [INSTANCE_OF_MODEL_PICKLE]
