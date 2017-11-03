import os
import pickle

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

from qanta import logging
from qanta.wikipedia.cached_wikipedia import CachedWikipedia
from qanta.datasets.quiz_bowl import QuizBowlDataset
from qanta.spark import create_spark_context
from qanta.preprocess import preprocess_dataset, tokenize_question
from qanta.guesser.abstract import AbstractGuesser
from qanta.guesser.nn import create_load_embeddings_function, convert_text_to_embeddings_indices, compute_n_classes
from qanta.torch import (
    BaseLogger, TerminateOnNaN, Tensorboard,
    EarlyStopping, ModelCheckpoint, MaxEpochStopping, TrainingManager, create_save_model
)


log = logging.get(__name__)


MEM_WE_TMP = '/tmp/qanta/deep/mem_we.pickle'
MEM_WE = 'mem_we.pickle'
load_embeddings = create_load_embeddings_function(MEM_WE_TMP, MEM_WE, log)


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


def search(text, n=10):
    s = Search(index='mem')[0:n].query('match', content=text)
    results = s.execute()
    memories = []
    for r in results:
        memories.append((r.meta.score, r.content, r.page))

    while len(memories) < n:
        memories.append(None)
    return memories


def load_memories(text_list, n):
    if os.path.exists('/tmp/memories.pickle'):
        with open('/tmp/memories.pickle', 'rb') as f:
            memory_lookup = pickle.load(f)
    else:
        memory_lookup = {}
    memory_size = len(memory_lookup)

    if memory_size == 0:
        # If everything is missing, then parallelize for speed
        sc = create_spark_context()
        memories = sc.parallelize(text_list, 256).map(lambda t: search(t, n=n)).collect()
        for text, mem in zip(text_list, memories):
            memory_lookup[text] = mem
    else:
        # If only some things are missing, use the cache and query what is missing
        memories = []
        for text in text_list:
            if text in memory_lookup:
                memories.append(memory_lookup[text])
            else:
                mem = search(text, n=n)
                memories.append(mem)
                memory_lookup[text] = mem

    if memory_size != len(memory_lookup):
        with open('/tmp/memories.pickle', 'wb') as f:
            pickle.dump(memory_lookup, f)

    return memories


def memories_to_indices(mems_list, embedding_lookup):
    all_key_indices = []
    all_value_classes = []
    all_scores = []
    for row in mems_list:
        # Each row contains a list of memories/text
        row_keys = []
        row_values = []
        row_scores = []
        for score, content, page in row:
            row_scores.append(score)
            # For each text in the row, convert it to embedding indices
            key_indices = convert_text_to_embeddings_indices(content, embedding_lookup)
            if len(key_indices) == 0:
                key_indices.append(embedding_lookup['UNK'])
            row_keys.append(key_indices)
            row_values.append(page)

        all_key_indices.append(row_keys)
        all_value_classes.append(row_values)
        all_scores.append(row_scores)

    return np.array(all_key_indices), all_value_classes, np.array(all_scores)


class KeyValueGuesser(AbstractGuesser):
    def __init__(self, n_memories=10):
        super().__init__()
        self.n_memories = n_memories
        self.class_to_i = None
        self.i_to_class = None
        self.vocab = None
        self.embeddings = None
        self.embedding_lookup = None
        self.n_classes = None

    def train(self, training_data):
        x_train_text, y_train, x_test_text, y_test, vocab, class_to_i, i_to_class = preprocess_dataset(
            training_data
        )
        self.class_to_i = class_to_i
        self.i_to_class = i_to_class
        self.vocab = vocab

        embeddings, embedding_lookup = load_embeddings(vocab=vocab, expand_glove=True)
        self.embeddings = embeddings
        self.embedding_lookup = embedding_lookup

        x_train = [convert_text_to_embeddings_indices(q, embedding_lookup) for q in x_train_text]
        for row in x_train:
            if len(row) == 0:
                row.append(embedding_lookup['UNK'])
        x_train = np.array(x_train)
        y_train = np.array(y_train)

        mems_train = load_memories([' '.join(x) for x in x_train_text], self.n_memories)
        mems_indices_train = memories_to_indices(mems_train, embedding_lookup)

        x_test = [convert_text_to_embeddings_indices(q, embedding_lookup) for q in x_test_text]
        for row in x_test:
            if len(row) == 0:
                row.append(embedding_lookup['UNK'])
        x_test = np.array(x_test)
        y_test = np.array(y_test)

        mems_test = load_memories([' '.join(x) for x in x_test_text], self.n_memories)
        mem_indices_test = memories_to_indices(mems_test, embedding_lookup)

        self.n_classes = compute_n_classes(training_data[1])

    def guess(self, questions, max_n_guesses):
        pass

    @classmethod
    def targets(cls):
        return []

    @classmethod
    def load(cls, directory):
        pass

    def save(self, directory):
        pass


class KeyValueNetwork(nn.Module):
    def __init__(self, vocab_size, n_classes, embedding_dim=300, dropout_prob=.3):
        super(KeyValueNetwork, self).__init__()
        self.vocab_size = vocab_size
        self.n_classes = n_classes
        self.embedding_dim = embedding_dim
        self.dropout_prob = dropout_prob

        self.dropout = nn.Dropout(dropout_prob)
        self.embeddings = nn.EmbeddingBag(vocab_size, embedding_dim)
        self.classification_layer = nn.Sequential(
            nn.Linear(embedding_dim, n_classes),
            nn.BatchNorm1d(n_classes),
            nn.Dropout(dropout_prob)
        )

    def init_weights(self, embeddings=None):
        if embeddings is not None:
            self.embeddings.weight = nn.Parameter(torch.from_numpy(embeddings).float())

    def forward(self):
        pass
