import pickle
import time
import os
import shutil
from typing import List, Tuple, Optional

from qanta.datasets.abstract import TrainingData
from qanta.datasets.quiz_bowl import QuizBowlDataset
from qanta.guesser.abstract import AbstractGuesser, QuestionText, Answer
from qanta.guesser.nn import (make_layer, convert_text_to_embeddings_indices, compute_lengths, compute_max_len,
                              tf_format, create_load_embeddings_function)
from qanta.wikipedia.cached_wikipedia import CachedWikipedia
from qanta.preprocess import preprocess_dataset, tokenize_question
from qanta.util.io import safe_open, safe_path, shell
from qanta.guesser.elasticsearch import ElasticSearchGuesser
from qanta.config import conf
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


def create_binarized_batches(
        batch_size,
        x_data: np.ndarray, x_lengths,
        wiki_data: np.ndarray, wiki_lengths: np.ndarray,
        neg_wiki_data: np.ndarray, neg_wiki_lengths: np.ndarray,
        pad=False, shuffle=True):
    if (type(x_data) != np.ndarray or
            type(x_lengths) != np.ndarray or
            type(wiki_data) != np.ndarray or
            type(wiki_lengths) != np.ndarray or
            type(neg_wiki_data) != np.ndarray or
            type(neg_wiki_lengths) != np.ndarray):
        log.info('type(x_data)={}'.format(x_data))
        log.info('type(x_lengths)={}'.format(x_lengths))
        log.info('type(wiki_data)={}'.format(wiki_data))
        log.info('type(wiki_lengths)={}'.format(wiki_lengths))
        log.info('type(neg_wiki_data)={}'.format(neg_wiki_data))
        log.info('type(neg_wiki_lengths)={}'.format(neg_wiki_lengths))
        raise ValueError('All inputs must be numpy arrays')

    if len({
        len(x_data), len(x_lengths),
        len(wiki_data), len(wiki_lengths),
        len(neg_wiki_data), len(neg_wiki_lengths)}) != 1:
        log.info('len(x_data)={}'.format(len(x_data)))
        log.info('len(x_lengths)={}'.format(len(x_lengths)))
        log.info('len(wiki_data)={}'.format(len(wiki_data)))
        log.info('len(wiki_lengths)={}'.format(len(wiki_lengths)))
        log.info('len(neg_wiki_data)={}'.format(len(neg_wiki_data)))
        log.info('len(neg_wiki_lengths)={}'.format(len(neg_wiki_lengths)))
        raise ValueError('All inputs must have the same length')
    n = len(x_data)
    order = list(range(n))
    if shuffle:
        np.random.shuffle(order)
    for i in range(0, n, batch_size):
        if len(order[i:i + batch_size]) == batch_size:
            x_batch = x_data[order[i:i + batch_size]]
            x_batch_lengths = x_lengths[order[i:i + batch_size]]
            wiki_batch = wiki_data[order[i:i + batch_size]]
            wiki_batch_lengths = wiki_lengths[order[i:i + batch_size]]
            neg_wiki_batch = neg_wiki_data[order[i:i + batch_size]]
            neg_wiki_batch_lengths = neg_wiki_lengths[order[i:i + batch_size]]
            yield x_batch, x_batch_lengths, wiki_batch, wiki_batch_lengths, neg_wiki_batch, neg_wiki_batch_lengths
        elif pad:
            size = len(order[i:i + batch_size])
            x_batch = np.vstack((
                x_data[order[i:i + batch_size]],
                np.zeros((batch_size - size, x_data.shape[1])))
            )
            x_batch_lengths = np.hstack((
                x_lengths[order[i:i + batch_size]],
                np.zeros((batch_size - size,)))
            )
            wiki_batch = np.vstack((
                wiki_data[order[i:i + batch_size]],
                np.zeros((batch_size - size, wiki_data.shape[1])))
            )
            wiki_batch_lengths = np.hstack((
                wiki_lengths[order[i:i + batch_size]],
                np.zeros((batch_size - size,)))
            )
            neg_wiki_batch = np.vstack((
                neg_wiki_data[order[i:i + batch_size]],
                np.zeros((batch_size - size, neg_wiki_data.shape[1])))
            )
            neg_wiki_batch_lengths = np.hstack((
                neg_wiki_lengths[order[i:i + batch_size]],
                np.zeros((batch_size - size,)))
            )
            yield x_batch, x_batch_lengths, wiki_batch, wiki_batch_lengths, neg_wiki_batch, neg_wiki_batch_lengths
        else:
            break


def create_wikipedia_batches(batch_size, wiki_data: np.ndarray, wiki_lengths: np.ndarray, pad=False, shuffle=True):
    if type(wiki_data) != np.ndarray or type(wiki_lengths) != np.ndarray:
        log.info('type(wiki_data)={}'.format(wiki_data))
        log.info('type(wiki_lengths)={}'.format(wiki_lengths))
        raise ValueError('All inputs must be numpy arrays')

    if len({len(wiki_data), len(wiki_lengths)}) != 1:
        log.info('len(wiki_data)={}'.format(len(wiki_data)))
        log.info('len(wiki_lengths)={}'.format(len(wiki_lengths)))
        raise ValueError('All inputs must have the same length')
    n = len(wiki_data)
    order = list(range(n))
    if shuffle:
        np.random.shuffle(order)
    for i in range(0, n, batch_size):
        if len(order[i:i + batch_size]) == batch_size:
            wiki_batch = wiki_data[order[i:i + batch_size]]
            wiki_batch_lengths = wiki_lengths[order[i:i + batch_size]]
            yield wiki_batch, wiki_batch_lengths
        elif pad:
            size = len(order[i:i + batch_size])
            wiki_batch = np.vstack((
                wiki_data[order[i:i + batch_size]],
                np.zeros((batch_size - size, wiki_data.shape[1])))
            )
            wiki_batch_lengths = np.hstack((
                wiki_lengths[order[i:i + batch_size]],
                np.zeros((batch_size - size,)))
            )
            yield wiki_batch, wiki_batch_lengths
        else:
            break


def create_test_batches(
        batch_size,
        x_data: np.ndarray, x_lengths, y_data: np.ndarray,
        wiki_dan_output: np.ndarray,
        pad=False, shuffle=True):
    if (type(x_data) != np.ndarray or
            type(x_lengths) != np.ndarray or
            type(y_data) != np.ndarray or
            type(wiki_dan_output) != np.ndarray):
        log.info('type(x_data)={}'.format(x_data))
        log.info('type(x_lengths)={}'.format(x_lengths))
        log.info('type(y_data)={}'.format(y_data))
        log.info('type(wiki_dan_output)={}'.format(wiki_dan_output))
        raise ValueError('All inputs must be numpy arrays')

    if len({len(x_data), len(x_lengths), len(y_data), len(wiki_dan_output)}) != 1:
        log.info('len(x_data)={}'.format(len(x_data)))
        log.info('len(x_lengths)={}'.format(len(x_lengths)))
        log.info('len(y_data)={}'.format(len(y_data)))
        log.info('len(wiki_dan_output)={}'.format(len(wiki_dan_output)))
        raise ValueError('All inputs must have the same length')
    n = len(x_data)
    order = list(range(n))
    if shuffle:
        np.random.shuffle(order)
    for i in range(0, n, batch_size):
        if len(order[i:i + batch_size]) == batch_size:
            x_batch = x_data[order[i:i + batch_size]]
            x_batch_lengths = x_lengths[order[i:i + batch_size]]
            y_batch = y_data[order[i:i + batch_size]]
            wiki_batch = wiki_dan_output[order[i:i + batch_size]]
            yield x_batch, x_batch_lengths, y_batch, wiki_batch
        elif pad:
            size = len(order[i:i + batch_size])
            x_batch = np.vstack((
                x_data[order[i:i + batch_size]],
                np.zeros((batch_size - size, x_data.shape[1])))
            )
            x_batch_lengths = np.hstack((
                x_lengths[order[i:i + batch_size]],
                np.zeros((batch_size - size,)))
            )
            y_batch = np.hstack((
                y_data[order[i:i + batch_size]],
                np.zeros((batch_size - size,)))
            )
            wiki_batch = np.vstack((
                wiki_dan_output[order[i:i + batch_size]],
                np.zeros((batch_size - size, wiki_dan_output.shape[1])))
            )
            yield x_batch, x_batch_lengths, y_batch, wiki_batch
        else:
            break


class DAN:
    def __init__(self, name,
                 text_placeholder, length_placeholder,
                 word_dropout_keep_prob, nn_dropout_keep_prob, is_training,
                 embeddings, max_input_length,
                 n_layers, n_hidden_units, reuse=None):
        self.n_layers = n_layers
        self.n_hidden_units = n_hidden_units
        self.max_input_length = max_input_length
        self._output = None
        with tf.variable_scope(name, reuse=reuse, initializer=tf.contrib.layers.xavier_initializer()):
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
                    dropout_prob=1 - nn_dropout_keep_prob,
                    batch_norm=True, batch_is_training=is_training,
                    reuse=reuse
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
                 max_n_epochs=100,
                 max_patience=5,
                 word_dropout_keep_prob=.5,
                 nn_dropout_keep_prob=conf['guessers']['BinarizedSiamese']['nn_dropout_keep_prob'],
                 class_to_i=None,
                 i_to_class=None,
                 wiki_pages=None,
                 wiki_length_map=None):
        self.session = None
        self.file_writer = None
        self.question_max_length = question_max_length
        self.wiki_max_length = wiki_max_length
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.n_hidden_units = n_hidden_units
        self.max_n_epochs = max_n_epochs
        self.max_patience = max_patience
        self.word_dropout_keep_prob = word_dropout_keep_prob
        self.nn_dropout_keep_prob = nn_dropout_keep_prob
        self.accuracy = None
        self.loss = None
        self.train_op = None
        self.probabilities = None
        self.summary_counter = 0
        self.wiki_pages = wiki_pages
        self.wiki_length_map = wiki_length_map
        self.class_to_i = class_to_i
        self.i_to_class = i_to_class
        self.cached_wikipedia = None
        self.question_wiki_similarity = None

        word_embeddings, word_embedding_lookup = load_embeddings()
        self.np_word_embeddings = word_embeddings
        self.embedding_lookup = word_embedding_lookup
        word_embeddings = tf.get_variable(
            'word_embeddings',
            initializer=tf.constant(word_embeddings, dtype=tf.float32)
        )
        self.word_embeddings = tf.pad(word_embeddings, [[0, 1], [0, 0]], mode='CONSTANT')

        self.word_dropout_keep_prob_var = tf.get_variable(
            'word_dropout_keep_prob', (), dtype=tf.float32, trainable=False)
        self.nn_dropout_keep_prob_var = tf.get_variable('nn_dropout_keep_prob', (), dtype=tf.float32, trainable=False)
        self.is_training = tf.placeholder(tf.bool, name='is_training')

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

        self.neg_wiki_data = tf.placeholder(
            tf.int32,
            shape=(self.batch_size, self.wiki_max_length),
            name='neg_wiki_data_input'
        )
        self.neg_wiki_lengths = tf.placeholder(tf.float32, shape=self.batch_size, name='neg_wiki_data_lengths')

        self.question_dan = DAN(
            'question_dan', self.qb_questions, self.question_lengths,
            self.word_dropout_keep_prob_var, self.nn_dropout_keep_prob_var, self.is_training,
            word_embeddings, self.question_max_length, self.n_layers, self.n_hidden_units
        )

        self.wiki_dan = DAN(
            'wiki_dan', self.wiki_data, self.wiki_lengths,
            self.word_dropout_keep_prob_var, self.nn_dropout_keep_prob_var, self.is_training,
            word_embeddings, self.wiki_max_length, self.n_layers, self.n_hidden_units
        )

        self.neg_wiki_dan = DAN(
            'wiki_dan', self.neg_wiki_data, self.neg_wiki_lengths,
            self.word_dropout_keep_prob_var, self.nn_dropout_keep_prob_var, self.is_training,
            word_embeddings, self.wiki_max_length, self.n_layers, self.n_hidden_units, reuse=True
        )

        with tf.variable_scope('similarity_prediction', reuse=None, initializer=tf.contrib.layers.xavier_initializer()):
            self.question_wiki_similarity = tf.reduce_sum(
                self.question_dan.output * self.wiki_dan.output, axis=1)

            self.question_neg_wiki_similarity = tf.reduce_sum(
                self.question_dan.output * self.neg_wiki_dan.output, axis=1)

        with tf.name_scope('metrics'):
            self.loss = tf.reduce_sum(
                tf.maximum(0.0, .1 - self.question_wiki_similarity + self.question_neg_wiki_similarity)
            )
            tf.summary.scalar('rank_loss', self.loss)

            self.accuracy = tf.reduce_mean(tf.cast(
                tf.greater_equal(self.question_wiki_similarity - self.question_neg_wiki_similarity, 0.0),
                tf.float32
            ))
            tf.summary.scalar('accuracy', self.accuracy)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer()
            self.train_op = optimizer.minimize(self.loss)

        self.summary = tf.summary.merge_all()
        self.saver = tf.train.Saver()

    def train(self, session, file_writer,
              x_train, y_train, x_train_lengths,
              x_test, y_test, x_test_lengths):
        self.session = session
        self.file_writer = file_writer
        self.wiki_pages = {}
        self.wiki_length_map = {}
        self.cached_wikipedia = CachedWikipedia()
        classes = set(y_train) | set(y_test)
        for c_index in classes:
            page = self.i_to_class[c_index]
            tokens = word_tokenize(self.cached_wikipedia[page].content[0:3000].strip().lower())
            w_indices = convert_text_to_embeddings_indices(tokens, self.embedding_lookup)
            self.wiki_length_map[c_index] = max(1, min(len(w_indices), self.wiki_max_length))
            while len(w_indices) < self.wiki_max_length:
                w_indices.append(self.np_word_embeddings.shape[0])
            w_indices = np.array(w_indices[:self.wiki_max_length])
            self.wiki_pages[c_index] = w_indices
        wiki_train = []
        wiki_train_lengths = []
        wiki_test = []
        wiki_test_lengths = []

        for y in y_train:
            wiki_train.append(self.wiki_pages[y])
            wiki_train_lengths.append(self.wiki_length_map[y])

        for y in y_test:
            wiki_test.append(self.wiki_pages[y])
            wiki_test_lengths.append(self.wiki_length_map[y])

        wiki_train = np.array(wiki_train)
        wiki_train_lengths = np.array(wiki_train_lengths)

        wiki_test = np.array(wiki_test)
        wiki_test_lengths = np.array(wiki_test_lengths)

        max_accuracy = -1
        patience = 0

        train_accuracies = []
        train_losses = []
        train_runtimes = []

        validation_accuracies = []
        validation_losses = []
        validation_runtimes = []

        for i in range(self.max_n_epochs):
            train_accuracy, train_loss, train_runtime = self.run_epoch(
                x_train, x_train_lengths, y_train, wiki_train, wiki_train_lengths, True)
            train_accuracies.append(train_accuracy)
            train_losses.append(train_loss)
            train_runtimes.append(train_runtime)

            log.info('Train Epoch: {} Avg loss: {:.4f} Accuracy: {:.4f} Runtime: {:.2f} seconds'.format(
                i, train_loss, train_accuracy, train_runtime
            ))

            val_accuracy, val_loss, val_runtime = self.run_epoch(
                x_test, x_test_lengths, y_test, wiki_test, wiki_test_lengths, False)
            validation_accuracies.append(val_accuracy)
            validation_losses.append(val_loss)
            validation_runtimes.append(val_runtime)

            log.info('Val Epoch: {} Avg loss: {:.4f} Accuracy: {:.4f} Runtime: {:.2f} seconds'.format(
                i, val_loss, val_accuracy, val_runtime
            ))

            patience += 1

            if val_accuracy > max_accuracy:
                log.info('New Best Accuracy, saving model')
                max_accuracy = val_accuracy
                patience = 0
                self.save()
            elif patience == self.max_patience:
                break

    def run_epoch(self, x_data, x_lengths, y_data, wiki_data, wiki_data_lengths, is_train):
        start_time = time.time()
        batch_accuracies = []
        batch_losses = []
        batch_i = 0

        if is_train:
            fetches = self.loss, self.accuracy, self.train_op
        else:
            fetches = self.loss, self.accuracy, self.summary

        self.session.run(self.word_dropout_keep_prob_var.assign(self.word_dropout_keep_prob if is_train else 1))
        self.session.run(self.nn_dropout_keep_prob_var.assign(self.nn_dropout_keep_prob if is_train else 1))
        neg_wiki_data = np.copy(wiki_data)
        np.random.shuffle(neg_wiki_data)
        neg_wiki_data_lengths = np.copy(wiki_data_lengths)
        np.random.shuffle(neg_wiki_data_lengths)

        batch_iter = create_binarized_batches(
            self.batch_size,
            x_data, x_lengths,
            wiki_data, wiki_data_lengths,
            neg_wiki_data, neg_wiki_data_lengths
        )

        for x_batch, x_len_batch, wiki_batch, wiki_len_batch, neg_wiki_batch, neg_wiki_len_batch in batch_iter:
            feed_dict = {
                self.qb_questions: x_batch,
                self.question_lengths: x_len_batch,
                self.wiki_data: wiki_batch,
                self.wiki_lengths: wiki_len_batch,
                self.neg_wiki_data: neg_wiki_batch,
                self.neg_wiki_lengths: neg_wiki_len_batch,
                self.is_training: int(is_train)
            }
            returned = self.session.run(fetches, feed_dict=feed_dict)
            loss = returned[0]
            accuracy = returned[1]
            if not is_train:
                summary = returned[2]
                self.file_writer.add_summary(summary, self.summary_counter)
                self.summary_counter += 1

            batch_losses.append(loss)
            batch_accuracies.append(accuracy)
            batch_i += 1

        runtime = time.time() - start_time

        return np.mean(batch_accuracies), np.mean(batch_losses), runtime

    def guess(self, candidates: List[List[str]], x_test, x_test_lengths, n_guesses: Optional[int]):
        log.info('Initializing tensorflow')
        self.session.run(tf.global_variables_initializer())
        self.load()
        self.session.run(self.word_dropout_keep_prob_var.assign(1))
        self.session.run(self.nn_dropout_keep_prob_var.assign(1))

        n_candidates_per_question = []
        string_candidates = []
        candidate_indices = []
        all_x_test = []
        all_x_test_lengths = []
        for i, question_candidates in enumerate(candidates):
            qb_words = x_test[i]
            qb_length = x_test_lengths[i]
            n_candidates_for_q = 0
            for string_c in question_candidates:
                if string_c in self.class_to_i:
                    c_index = self.class_to_i[string_c]
                    string_candidates.append(string_c)
                    candidate_indices.append(c_index)
                    all_x_test.append(qb_words)
                    all_x_test_lengths.append(qb_length)
                    n_candidates_for_q += 1
            n_candidates_per_question.append(n_candidates_for_q)
        all_x_test = np.array(all_x_test)
        all_x_test_lengths = np.array(all_x_test_lengths)
        n_candidates = len(candidates)

        all_wiki = []
        all_wiki_lengths = []

        n_wiki_candidates = len(self.i_to_class)
        for i in range(n_wiki_candidates):
            all_wiki.append(self.wiki_pages[i])
            all_wiki_lengths.append(self.wiki_length_map[i])

        all_wiki = np.array(all_wiki)
        all_wiki_lengths = np.array(all_wiki_lengths)

        wiki_dan_output = []
        for wiki_batch, wiki_len_batch in create_wikipedia_batches(self.batch_size, all_wiki, all_wiki_lengths,
                                                                   pad=True, shuffle=False):
            feed_dict = {
                self.wiki_data: wiki_batch,
                self.wiki_lengths: wiki_len_batch,
                self.is_training: 0
            }
            wiki_dan_output.append(self.session.run(self.wiki_dan.output, feed_dict=feed_dict))
        wiki_dan_output = np.vstack(wiki_dan_output)[:n_wiki_candidates]

        all_x_wiki_dan = []
        for c_index in candidate_indices:
            all_x_wiki_dan.append(wiki_dan_output[c_index])
        all_x_wiki_dan = np.array(all_x_wiki_dan)

        n = len(x_test)
        log.info(
            'Scoring each of {} questions totalling {} scored guesses across all questions'.format(n, n_candidates))
        all_guesses = []
        y_labels = np.zeros((n_candidates,))
        all_predictions = []

        for x_batch, x_len_batch, y_batch, wiki_dan_batch in create_test_batches(
                self.batch_size, all_x_test, all_x_test_lengths, y_labels, all_x_wiki_dan,
                pad=True, shuffle=False):
            feed_dict = {
                self.qb_questions: x_batch,
                self.question_lengths: x_len_batch,
                self.wiki_dan.output: wiki_dan_batch,
                self.is_training: 0
            }
            batch_predictions = self.session.run(self.question_wiki_similarity, feed_dict=feed_dict)
            all_predictions.extend(batch_predictions)

        scores = all_predictions[:n_candidates]
        guesses_with_scores = list(zip(string_candidates, scores))

        position = 0
        for n in n_candidates_per_question:
            candidate_guesses = sorted(guesses_with_scores[position:position + n], reverse=True, key=lambda gs: gs[1])
            all_guesses.append(candidate_guesses[:n_guesses])
            position += n

        return all_guesses

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
        self.wiki_pages = None
        self.wiki_length_map = None

    def qb_dataset(self):
        return QuizBowlDataset(2)

    @classmethod
    def targets(cls) -> List[str]:
        return [BINARIZED_PARAMS_TARGET]

    def train(self, training_data: TrainingData) -> None:
        log.info('Preprocessing training data...')
        x_train, y_train, _, x_test, y_test, _, vocab, class_to_i, i_to_class = preprocess_dataset(training_data)
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
            self.model = BinarizedSiameseModel(
                self.question_max_length, 550,
                i_to_class=self.i_to_class, class_to_i=self.class_to_i
            )
            session.run(tf.global_variables_initializer())
            self.file_writer = tf.summary.FileWriter(os.path.join('output/tensorflow', 'binarized_logs'), session.graph)
            self.model.train(
                session, self.file_writer,
                x_train, y_train, x_train_lengths,
                x_test, y_test, x_test_lengths
            )
            self.wiki_pages = self.model.wiki_pages
            self.wiki_length_map = self.model.wiki_length_map

    def guess(self, questions: List[QuestionText], max_n_guesses: Optional[int]) -> List[List[Tuple[Answer, float]]]:
        log.info('Generating {} guesses for each of {} questions'.format(max_n_guesses, len(questions)))
        candidates_with_scores = ElasticSearchGuesser().guess(questions, 200)
        candidates = []
        for question_candidates in candidates_with_scores:
            candidates.append([candidate for candidate, _ in question_candidates])
        x_test = [convert_text_to_embeddings_indices(tokenize_question(q), self.embedding_lookup) for q in questions]
        x_test_lengths = compute_lengths(x_test)
        x_test = np.array(tf_format(x_test, self.question_max_length, self.embeddings.shape[0]))

        with tf.Graph().as_default(), tf.Session() as session:
            self.model = BinarizedSiameseModel(
                self.question_max_length, 550,
                i_to_class=self.i_to_class, class_to_i=self.class_to_i,
                wiki_pages=self.wiki_pages
            )
            self.model.session = session
            return self.model.guess(candidates, x_test, x_test_lengths, max_n_guesses)

    def save(self, directory: str) -> None:
        params_path = os.path.join(directory, BINARIZED_PARAMS_TARGET)
        with safe_open(params_path, 'wb') as f:
            pickle.dump({
                'class_to_i': self.class_to_i,
                'i_to_class': self.i_to_class,
                'vocab': self.vocab,
                'question_max_length': self.question_max_length,
                'wiki_pages': self.wiki_pages,
                'wiki_length_map': self.wiki_length_map
            }, f)
            model_path = os.path.join(directory, BINARIZED_MODEL_TARGET)
            shell('cp -r {} {}'.format(BINARIZED_MODEL_TMP_DIR, safe_path(model_path)))
            we_path = os.path.join(directory, BINARIZED_WE)
            shutil.copyfile(BINARIZED_WE_TMP, safe_path(we_path))

    @classmethod
    def load(cls, directory: str):
        guesser = BinarizedGuesser()
        embeddings, embedding_lookup = load_embeddings(root_directory=directory)
        params_path = os.path.join(directory, BINARIZED_PARAMS_TARGET)
        with open(params_path, 'rb') as f:
            params = pickle.load(f)
            guesser.class_to_i = params['class_to_i']
            guesser.i_to_class = params['i_to_class']
            guesser.vocab = params['vocab']
            guesser.question_max_length = params['question_max_length']
            guesser.wiki_pages = params['wiki_pages']
            guesser.wiki_length_map = params['wiki_length_map']
            guesser.embeddings = embeddings
            guesser.embedding_lookup = embedding_lookup

        model_path = os.path.join(directory, BINARIZED_MODEL_TARGET)
        shell('cp -r {} {}'.format(model_path, safe_path(BINARIZED_MODEL_TMP_DIR)))
        we_path = os.path.join(directory, BINARIZED_WE)
        shutil.copyfile(BINARIZED_WE_TMP, we_path)
        return guesser
