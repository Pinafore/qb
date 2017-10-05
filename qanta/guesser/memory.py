from elasticsearch_dsl import DocType, Text, Keyword, Search, Index
from elasticsearch_dsl.connections import connections
import elasticsearch

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.optim import Adam, lr_scheduler

import progressbar

from qanta.wikipedia.cached_wikipedia import CachedWikipedia
from qanta.datasets.quiz_bowl import QuizBowlDataset
from qanta.spark import create_spark_context
from qanta.preprocess import preprocess_dataset, tokenize_question
from qanta.guesser.abstract import AbstractGuesser
from qanta.guesser.nn import create_load_embeddings_function, convert_text_to_embeddings_indices, compute_n_classes
from qanta.manager import (
    BaseLogger, TerminateOnNaN, Tensorboard,
    EarlyStopping, ModelCheckpoint, MaxEpochStopping, TrainingManager
)
from qanta.guesser.torch.util import create_save_model



connections.create_connection(hosts='localhost')


class Answer(DocType):
    page = Text(fields={'raw': Keyword()})
    content = Text()

    class Meta:
        index = 'mem'


def paragraph_tokenize(page):
    # The first element is always just the wikipedia page title
    return [c for c in page.content.split('\n') if c != ''][1:]


def index_page(wiki_page):
    page = wiki_page.title
    for paragraph in paragraph_tokenize(wiki_page):
        Answer(page=page, content=paragraph).save()


def create_memory_index():
    dataset = QuizBowlDataset(guesser_train=True)
    training_data = dataset.training_data()
    answers = set(training_data[1])
    cw = CachedWikipedia()

    try:
        Index('mem').delete()
    except:
        pass
    Answer.init()
    all_wiki_pages = [cw[page] for page in answers]
    wiki_pages = [p for p in all_wiki_pages if p.content != '']
    sc = create_spark_context()
    sc.parallelize(wiki_pages, 1000).foreach(index_page)


def search(text):
    s = Search(index='mem')[0:50].query('match', content=text)
    results = s.execute()
    memories = []
    for r in results:
        memories.append((r.page, r.content, r.meta.score))
    return memories

