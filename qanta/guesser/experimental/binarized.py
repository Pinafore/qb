import pickle
import time
import os
import shutil
from typing import List, Tuple, Optional

from qanta.datasets.abstract import TrainingData
from qanta.datasets.quiz_bowl import QuizBowlDataset
from qanta.guesser.abstract import AbstractGuesser, QuestionText, Answer
from qanta.guesser.nn import (make_layer, convert_text_to_embeddings_indices, compute_n_classes,
                              compute_lengths, compute_max_len, tf_format,
                              create_load_embeddings_function, batch_iterables)
from qanta.wikipedia.cached_wikipedia import CachedWikipedia
from qanta.preprocess import preprocess_dataset, tokenize_question
from qanta.util.io import safe_open, safe_path, shell
from qanta import logging

import tensorflow as tf
import numpy as np
from nltk import word_tokenize

log = logging.get(__name__)
BINARIZED_WE_TMP = '/tmp/qanta/deep/binarized_we.pickle'
BINARIZED_WE = 'binarized_we.pickle'
GLOVE_WE = 'data/external/deep/glove.6B.300d.txt'
BINARIZED_MODEL_TMP_PREFIX = '/tmp/qanta/deep/binarized'
BINARIZED_MODEL_TMP_DIR = '/tmp/qanta/deep'
BINARIZED_MODEL_TARGET = 'binarized_dir'
BINARIZED_PARAMS_TARGET = 'binarized_params.pickle'


load_embeddings = create_load_embeddings_function(BINARIZED_WE_TMP, BINARIZED_WE, log)


class DAN:
    def __init__(self, name,
                 text_placeholder, length_placeholder,
                 word_dropout_keep_prob, nn_dropout_keep_prob, is_training,
                 embeddings, max_input_length,
                 n_layers, n_hidden_units):
        self.n_layers = n_layers
        self.n_hidden_units = n_hidden_units
        self.max_input_length = max_input_length
        self._output = None
        with tf.variable_scope(name, reuse=None, initializer=tf.contrib.layers.xaviar_initializer()):
            word_vectors = tf.nn.embedding_lookup(embeddings, text_placeholder)

            word_drop_filter = tf.nn.dropout(tf.ones((self.max_input_length, 1)), keep_prob=word_dropout_keep_prob)
            self.word_vectors = word_vectors * word_drop_filter
            self.avg_word_vectors = tf.reduce_sum(self.word_vectors, 1) / tf.expand_dims(
                length_placeholder, 1)

            in_dim = embeddings.get_shape()[1]
            layer_out = self.avg_word_vectors
            for i in range(self.n_layers):
                layer_out, _ = make_layer(
                    i, layer_out,
                    n_in=in_dim, n_out=self.n_hidden_units, op=tf.nn.elu,
                    dropout_prob=1-nn_dropout_keep_prob,
                    batch_norm=True, batch_is_training=is_training
                )
                in_dim = None
            self._output = layer_out

    @property
    def output(self):
        return self._output


class BinarizedSiameseModel:
    def __init__(self,
                 question_max_length,
                 wiki_max_length,
                 batch_size=128,
                 n_layers=1,
                 n_hidden_units=200,
                 max_n_epochs=100):
        self.session = None
        self.question_max_length = question_max_length
        self.wiki_max_length = wiki_max_length
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.n_hidden_units = n_hidden_units
        self.max_n_epochs = max_n_epochs

        word_embeddings, word_embedding_lookup = load_embeddings()
        self.embedding_lookup = word_embedding_lookup
        word_embeddings = tf.get_variable(
            'word_embeddings',
            tf.constant(word_embeddings, dtype=tf.float32)
        )
        self.word_embeddings = tf.pad(self.word_embeddings, [[0, 1], [0, 0]], mode='CONSTANT')

        self.word_dropout_keep_prob = tf.get_variable(
            'word_dropout_keep_prob', (), dtype=tf.float32, trainable=False)
        self.nn_dropout_keep_prob = tf.get_variable('nn_dropout_keep_prob', (), dtype=tf.float32, trainable=False)
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.labels = tf.placeholder(tf.int32, shape=self.batch_size, name='labels')

        self.qb_questions = tf.placeholder(
            tf.int32,
            shape=(self.batch_size, self.question_max_length),
            name='question_input'
        )
        self.question_lengths = tf.placeholder(tf.float32, shape=self.batch_size, name='question_lengths')

        self.wiki_data = tf.placeholder(
            tf.int32,
            shape=(self.batch_size, self.wiki_max_length),
            name='wiki_data_input'
        )
        self.wiki_lengths = tf.placeholder(tf.float32, shape=self.batch_size, name='wiki_data_lengths')

        self.question_dan = DAN(
            'question_dan', self.qb_questions, self.question_lengths,
            self.word_dropout_keep_prob, self.nn_dropout_keep_prob, self.is_training,
            word_embeddings, self.question_max_length, self.n_layers, self.n_hidden_units
        )

        self.wiki_dan = DAN(
            'wiki_dan', self.wiki_data, self.wiki_lengths,
            self.word_dropout_keep_prob, self.nn_dropout_keep_prob, self.is_training,
            word_embeddings, self.wiki_max_length, self.n_layers, self.n_hidden_units
        )

        with tf.variable_scope('similarity_prediction', reuse=None, initializer=tf.contrib.layers.xaviar_initializer()):
            self.question_wiki_similarity = self.question_dan.output * self.wiki_dan.output
            self.probabilities, _ = make_layer(-1, self.question_wiki_similarity, n_out=1, op=tf.nn.sigmoid)
            self.probabilities = tf.reshape(self.probabilities, (-1,))
            self.predictions = tf.round(self.probabilities)

        with tf.name_scope('metrics'):
            self.loss = tf.losses.log_loss(self.labels, self.probabilities)
            self.loss = tf.reduce_mean(self.loss)
            tf.summary.scalar('log_loss', self.loss)

            self.accuracy = tf.metrics.accuracy(self.labels, self.predictions)
            tf.summary.scalar('accuracy', self.accuracy)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer()
            self.train_op = optimizer.minimize(self.loss)

        self.summary = tf.summary.merge_all()
        self.saver = tf.train.Saver()

    def train(self, session,
              x_train, y_train, x_train_lengths,
              x_test, y_test, x_test_lengths,
              class_to_i, i_to_class):
        wiki_pages = {}
        cw = CachedWikipedia()
        classes = set(y_train) | set(y_test)
        for c_index in classes:
            page = i_to_class[c_index]
            tokens = word_tokenize(cw[page].content[0:3000].strip().lower())
            w_indices = convert_text_to_embeddings_indices(tokens, self.embedding_lookup)
            wiki_pages[c_index] = w_indices

        wiki_train = []
        wiki_test = []
        negative_y_train = np.copy(y_train)
        np.random.shuffle(negative_y_train)
        negative_y_test = np.copy(y_test)
        np.random.shuffle(negative_y_test)
        n_train_examples = len(x_train)
        n_test_examples = len(x_test)

        for y in y_train:
            wiki_train.append(wiki_pages[y])

        for y in negative_y_train:
            wiki_train.append(wiki_pages[y])

        for y in y_test:
            wiki_test.append(wiki_pages[y])

        for y in negative_y_test:
            wiki_test.append(wiki_pages[y])

        wiki_train = np.array(wiki_train)
        all_x_train = np.vstack([x_train, x_train])
        all_y_train = np.vstack([np.ones((n_train_examples,)), np.zeros((n_train_examples,))])

        wiki_test = np.array(wiki_test)
        all_x_test = np.vstack([x_test, x_test])
        all_y_test = np.vstack([np.ones((n_test_examples,)), np.zeros((n_test_examples,))])



    def save(self):
        self.saver.save(self.session, safe_path(BINARIZED_MODEL_TMP_PREFIX))

    def load(self):
        self.saver.restore(self.session, BINARIZED_MODEL_TMP_PREFIX)


class BinarizedGuesser(AbstractGuesser):
    def __init__(self):
        super().__init__()
        self.class_to_i = None
        self.i_to_class = None
        self.vocab = None
        self.embeddings = None
        self.embedding_lookup = None
        self.model = None
        self.question_max_length = None
        self.file_writer = None

    def qb_dataset(self):
        return QuizBowlDataset(1)

    @classmethod
    def targets(cls) -> List[str]:
        return [BINARIZED_PARAMS_TARGET]

    def train(self, training_data: TrainingData) -> None:
        log.info('Preprocessing training data...')
        x_train, y_train, _, x_test, y_test, _, vocab, class_to_i, i_to_class = preprocess_dataset(training_data)
        n_train_examples = len(x_train)
        n_test_examples = len(x_test)
        self.class_to_i = class_to_i
        self.i_to_class = i_to_class
        self.vocab = vocab
        embeddings, embedding_lookup = load_embeddings(vocab=vocab)
        self.embeddings = embeddings
        self.embedding_lookup = embedding_lookup

        x_train = [convert_text_to_embeddings_indices(q, embedding_lookup) for q in x_train]
        x_train_lengths = compute_lengths(x_train)

        x_test = [convert_text_to_embeddings_indices(q, embedding_lookup) for q in x_test]
        x_test_lengths = compute_lengths(x_test)

        self.question_max_length = compute_max_len(training_data)

        x_train = tf_format(x_train, self.question_max_length, embeddings.shape[0])
        x_test = tf_format(x_test, self.question_max_length, embeddings.shape[0])
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        with tf.Graph().as_default(), tf.Session() as session:
            self.model = BinarizedSiameseModel(self.question_max_length, 550)
            session.run(tf.global_variables_initializer())
            self.file_writer = tf.summary.FileWriter(os.path.join('output/tensorflow', 'binarized_logs', session.graph))
            self.model.train(
                session,
                x_train, y_train, x_train_lengths,
                x_test, y_test, x_test_lengths,
                class_to_i, i_to_class
            )

    def save(self, directory: str) -> None:
        pass

    @classmethod
    def load(cls, directory: str):
        pass

    def guess(self, questions: List[QuestionText], max_n_guesses: Optional[int]) -> List[List[Tuple[Answer, float]]]:
        pass

