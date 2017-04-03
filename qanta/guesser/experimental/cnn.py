"""
This file implements a TF based fixed length CNN neural guesser. It takes a fixed length number of input
words and attempts to predict the answer
"""
import pickle
import time
import os
import shutil
from typing import Dict, List, Tuple, Union, Set, Optional

from qanta.datasets.abstract import TrainingData
from qanta.guesser.abstract import AbstractGuesser
from qanta.datasets.quiz_bowl import QuizBowlDataset
from qanta.datasets.wikipedia import WikipediaDataset
from qanta.preprocess import preprocess_dataset, tokenize_question
from qanta.util.io import safe_open, safe_path, shell
from qanta import logging

import tensorflow as tf
import numpy as np

log = logging.get(__name__)
TF_DAN_WE_TMP = '/tmp/qanta/deep/cnn.pickle'
TF_DAN_WE = 'cnn.pickle'
GLOVE_WE = 'data/external/deep/glove.6B.300d.txt'
DEEP_CNN_MODEL_TMP_PREFIX = '/tmp/qanta/deep/cnn'
DEEP_CNN_MODEL_TMP_DIR = '/tmp/qanta/deep'
DEEP_CNN_MODEL_TARGET = 'cnn_dir'
DEEP_CNN_PARAMS_TARGET = 'cnn_params.pickle'


def _make_layer(i: int, in_tensor, n_out, op,
                n_in=None, dropout_prob=None, batch_norm=False, batch_is_training=None):
    with tf.variable_scope('layer' + str(i)):
        if batch_norm and batch_is_training is None:
            raise ValueError('if using batch norm then passing a training placeholder is required')
        w = tf.get_variable('w', (in_tensor.get_shape()[1] if n_in is None else n_in, n_out),
                            dtype=tf.float32)
        if dropout_prob is not None:
            w = tf.nn.dropout(w, keep_prob=1 - dropout_prob)
        b = tf.get_variable('b', n_out, dtype=tf.float32)
        out = tf.matmul(in_tensor, w) + b
        if batch_norm:
            out = tf.contrib.layers.batch_norm(
                out, center=True, scale=True, is_training=batch_is_training, scope='bn', fused=True)
        out = (out if op is None else op(out))
        return out, w


def _create_embeddings(vocab: Set[str]):
    embeddings = []
    embedding_lookup = {}
    with open(GLOVE_WE) as f:
        i = 0
        for l in f:
            splits = l.split()
            word = splits[0]
            if word in vocab:
                emb = [float(n) for n in splits[1:]]
                embeddings.append(emb)
                embedding_lookup[word] = i
                i += 1
        embeddings = np.array(embeddings)
        mean_embedding = embeddings.mean(axis=0)
        embed_with_unk = np.vstack([embeddings, mean_embedding])
        embedding_lookup['UNK'] = i
        return embed_with_unk, embedding_lookup


def _load_embeddings(vocab=None, root_directory=''):
    if os.path.exists(TF_DAN_WE_TMP):
        log.info('Loading word embeddings from tmp cache')
        with safe_open(TF_DAN_WE_TMP, 'rb') as f:
            return pickle.load(f)
    elif os.path.exists(os.path.join(root_directory, TF_DAN_WE)):
        log.info('Loading word embeddings from restored cache')
        with safe_open(os.path.join(root_directory, TF_DAN_WE), 'rb') as f:
            return pickle.load(f)
    else:
        if vocab is None:
            raise ValueError('To create fresh embeddings a vocab is needed')
        with safe_open(TF_DAN_WE_TMP, 'wb') as f:
            log.info('Creating word embeddings and saving to cache')
            embed_and_lookup = _create_embeddings(vocab)
            pickle.dump(embed_and_lookup, f)
            return embed_and_lookup


def _convert_text_to_embeddings_indices(words: List[str], embedding_lookup: Dict[str, int]):
    w_indices = []
    for w in words:
        if w in embedding_lookup:
            w_indices.append(embedding_lookup[w])
        else:
            w_indices.append(embedding_lookup['UNK'])
    return w_indices


def _compute_n_classes(labels: List[str]):
    return len(set(labels))


def _compute_max_len(x_data: List[List[int]]):
    return max(len(x) for x in x_data)


def _tf_format(x_data: List[List[int]], max_len: int, zero_index: int):
    """
    Pad with elements until it has max_len or shorten it until it has max_len. When padding insert
    the zero index so it doesn't contribute anything
    :param x_data:
    :param max_len:
    :return:
    """
    for i in range(len(x_data)):
        row = x_data[i]
        while len(row) < max_len:
            row.append(zero_index)
        x_data[i] = x_data[i][:max_len]
    return x_data


def _create_batches(batch_size,
                    x_data: np.ndarray, y_data: np.ndarray, x_lengths: np.ndarray,
                    pad=False, shuffle=True):
    if type(x_data) != np.ndarray or type(y_data) != np.ndarray:
        raise ValueError('x and y must be numpy arrays')
    if len(x_data) != len(y_data):
        raise ValueError('x and y must have the same dimension')
    n = len(x_data)
    order = list(range(n))
    if shuffle:
        np.random.shuffle(order)
    for i in range(0, n, batch_size):
        if len(order[i:i + batch_size]) == batch_size:
            x_batch = x_data[order[i:i + batch_size]]
            y_batch = y_data[order[i:i + batch_size]]
            x_batch_lengths = x_lengths[order[i:i + batch_size]]
            yield x_batch, y_batch, x_batch_lengths
        elif pad:
            size = len(order[i:i + batch_size])
            x_batch = np.vstack((
                x_data[order[i:i + batch_size]],
                np.zeros((batch_size - size, x_data.shape[1])))
            )
            y_batch = np.hstack((
                y_data[order[i:i + batch_size]],
                np.zeros((batch_size - size,)))
            )
            x_batch_lengths = np.hstack((
                x_lengths[order[i:i + batch_size]],
                np.zeros((batch_size - size,)))
            )
            yield x_batch, y_batch, x_batch_lengths
        else:
            break


def _compute_lengths(x_data):
    return np.array([max(1, len(x)) for x in x_data])


class CNNModel:
    def __init__(self, cnn_params: Dict, max_len: int, n_classes: int):
        self.cnn_params = cnn_params
        self.max_len = max_len
        self.n_classes = n_classes
        self.n_hidden_units = cnn_params['n_hidden_units']
        self.n_hidden_layers = cnn_params['n_hidden_layers']
        self.word_dropout = cnn_params['word_dropout']
        self.nn_dropout = cnn_params['nn_dropout']
        self.batch_size = cnn_params['batch_size']
        self.learning_rate = cnn_params['learning_rate']
        self.max_epochs = cnn_params['max_epochs']
        self.max_patience = cnn_params['max_patience']

        # These are set by build_tf_model
        self.input_placeholder = None
        self.len_placeholder = None
        self.label_placeholder = None
        self.loss = None
        self.batch_accuracy = None
        self.train_op = None
        self.softmax_output = None
        self.saver = None
        self.file_writer = None
        self.sent_vecs = None
        self.word_dropout_var = None
        self.nn_dropout_var = None
        self.initial_embed = None
        self.mean_embeddings = None
        self.embed_and_zero = None
        self.accuracy = None
        self.training_phase = None
        self.h_pool = None
        self.h_pool_flat = None
        self.h_drop = None

        # Set at runtime
        self.summary = None
        self.session = None
        self.summary_counter = 0
        self.filter_sizes = [3, 4, 5]
        self.num_filters = 128

    def build_tf_model(self):
        with tf.variable_scope(
                'dan',
                reuse=None,
                initializer=tf.contrib.layers.xavier_initializer()):
            embedding, embedding_word_lookup = _load_embeddings()
            self.initial_embed = tf.get_variable(
                'embedding',
                initializer=tf.constant(embedding, dtype=tf.float32)
            )
            embedding_size = embedding.shape[1]
            self.embed_and_zero = tf.pad(self.initial_embed, [[0, 1], [0, 0]], mode='CONSTANT')
            self.input_placeholder = tf.placeholder(
                tf.int32, shape=(self.batch_size, self.max_len), name='input_placeholder')
            self.len_placeholder = tf.placeholder(
                tf.float32, shape=self.batch_size, name='len_placeholder')
            self.label_placeholder = tf.placeholder(
                tf.int32, shape=self.batch_size, name='label_placeholder')
            self.training_phase = tf.placeholder(tf.bool, name='phase')

            # (batch_size, max_len, embedding_dim)
            self.sent_vecs = tf.nn.embedding_lookup(self.embed_and_zero, self.input_placeholder)
            self.word_dropout_var = tf.get_variable('word_dropout', (), dtype=tf.float32,
                                                    trainable=False)
            drop_filter = tf.nn.dropout(
                tf.ones((self.batch_size, self.max_len, 1)),
                keep_prob=1 - self.word_dropout_var)
            self.sent_vecs = self.sent_vecs * drop_filter
            self.sent_vecs = tf.expand_dims(self.sent_vecs, -1)

            self.nn_dropout_var = tf.get_variable('nn_dropout', (), dtype=tf.float32,
                                                  trainable=False)

            pooled_outputs = []
            for i, filter_size in enumerate(self.filter_sizes):
                with tf.name_scope('conv-maxpool-{}'.format(filter_size)):
                    filter_shape = [filter_size, embedding_size, 1, self.num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=.1), name='W')
                    b = tf.Variable(tf.constant(.1, shape=[self.num_filters]), name='b')
                    conv = tf.nn.conv2d(
                        self.sent_vecs,
                        W,
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name='conv'
                    )
                    h_bias_add = tf.nn.bias_add(conv, b)
                    h = tf.nn.relu(h_bias_add, name='relu')
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, self.max_len - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name='pool'
                    )
                    pooled_outputs.append(pooled)

            num_filters_total = self.num_filters * len(self.filter_sizes)
            self.h_pool = tf.concat(pooled_outputs, 3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
            with tf.name_scope('dropout'):
                self.h_drop = tf.nn.dropout(self.h_pool_flat, 1 - self.nn_dropout_var)

            with tf.name_scope('output'):
                W = tf.Variable(
                    tf.truncated_normal([num_filters_total, self.n_classes],
                                        stddev=.1), name='W')
                b = tf.Variable(tf.constant(.1, shape=[self.n_classes]), name='b')
                logits = tf.nn.xw_plus_b(self.h_drop, W, b, name='logits')
                predictions = tf.to_int32(tf.argmax(logits, 1, name='predictions'))
                self.softmax_output = tf.nn.softmax(logits)

            with tf.name_scope('loss'):
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits, labels=tf.to_int64(self.label_placeholder))
                self.loss = tf.reduce_mean(losses)
                tf.summary.scalar('loss', self.loss)

            with tf.name_scope('accuracy'):
                self.accuracy = tf.contrib.metrics.accuracy(
                    predictions, self.label_placeholder)
                tf.summary.scalar('accuracy', self.accuracy)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.name_scope('train'):
                with tf.control_dependencies(update_ops):
                    global_step = tf.Variable(0, name='global_step', trainable=False)
                    optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                    grads_and_vars = optimizer.compute_gradients(self.loss)
                    self.train_op = optimizer.apply_gradients(
                        grads_and_vars, global_step=global_step)

            self.summary = tf.summary.merge_all()
            self.saver = tf.train.Saver()

    def train(self, x_train, y_train, x_train_lengths, x_test, y_test, x_test_lengths, save=True):
        with tf.Graph().as_default(), tf.Session() as session:
            self.build_tf_model()
            self.session = session
            self.session.run(tf.global_variables_initializer())
            params_suffix = ','.join('{}={}'.format(k, v) for k, v in self.cnn_params.items())
            self.file_writer = tf.summary.FileWriter(
                os.path.join('output/tensorflow', params_suffix), session.graph)
            train_losses, train_accuracies, holdout_losses, holdout_accuracies = self._train(
                x_train, y_train, x_train_lengths,
                x_test, y_test, x_test_lengths,
                self.max_epochs, save=save
            )

            return train_losses, train_accuracies, holdout_losses, holdout_accuracies

    def _train(self,
               x_train, y_train, x_train_lengths,
               x_test, y_test, x_test_lengths,
               n_epochs: int, save=True):
        max_accuracy = -1
        patience = self.max_patience

        train_accuracies = []
        train_losses = []

        holdout_accuracies = []
        holdout_losses = []

        for i in range(n_epochs):
            # Training Epoch
            accuracies, losses, duration = self.run_epoch(
                x_train, y_train, x_train_lengths
            )
            log.info(
                'Train Epoch: {} Avg loss: {:.4f} Accuracy: {:.4f}. Ran in {:.4f} seconds.'.format(
                    i, np.average(losses), np.average(accuracies), duration))
            train_accuracies.append(accuracies)
            train_losses.append(losses)

            # Validation Epoch
            val_accuracies, val_losses, val_duration = self.run_epoch(
                x_test, y_test, x_test_lengths, train=False
            )
            val_accuracy = np.average(val_accuracies)
            log.info(
                'Val Epoch: {} Avg loss: {:.4f} Accuracy: {:.4f}. Ran in {:.4f} seconds.'.format(
                    i, np.average(val_losses), val_accuracy, val_duration))
            holdout_accuracies.append(val_accuracies)
            holdout_losses.append(val_losses)

            # Save the model if its better
            patience -= 1
            if val_accuracy > max_accuracy:
                max_accuracy = val_accuracy
                patience = self.max_patience
                if save:
                    log.info('New best accuracy, saving model')
                    self.save()
                else:
                    log.info('New best accuracy, model saving turned off')

            # Early stopping after some burn in
            if patience == 0:
                break

        return train_losses, train_accuracies, holdout_losses, holdout_accuracies

    def run_epoch(self, x_data, y_data, x_lengths, train=True):
        start_time = time.time()
        accuracies = []
        losses = []
        if train:
            fetches = self.loss, self.accuracy, self.train_op
        else:
            fetches = self.loss, self.accuracy, self.summary

        batch_i = 0
        self.session.run(self.word_dropout_var.assign(self.word_dropout if train else 0))
        self.session.run(self.nn_dropout_var.assign(self.nn_dropout if train else 0))
        for x_batch, y_batch, x_len_batch in _create_batches(
                self.batch_size, x_data, y_data, x_lengths):
            feed_dict = {
                self.input_placeholder: x_batch,
                self.label_placeholder: y_batch,
                self.len_placeholder: x_len_batch,
                self.training_phase: int(train)
            }
            returned = self.session.run(fetches, feed_dict=feed_dict)
            loss = returned[0]
            accuracy = returned[1]
            if not train:
                summary = returned[2]
                self.file_writer.add_summary(summary, self.summary_counter)
                self.summary_counter += 1

            accuracies.append(accuracy)
            losses.append(loss)
            batch_i += 1
        duration = time.time() - start_time
        return accuracies, losses, duration

    def guess(self, x_test, x_test_lengths, n_guesses: Optional[int]):
        with tf.Graph().as_default(), tf.Session() as session:
            self.build_tf_model()
            self.session = session
            self.session.run(tf.global_variables_initializer())
            self.load()
            y_test = np.zeros((x_test.shape[0]))
            self.session.run(self.word_dropout_var.assign(0))
            self.session.run(self.nn_dropout_var.assign(0))
            all_labels = []
            all_scores = []
            log.info('Starting dan tf batches...')
            batch_i = 0
            column_index = [[i] for i in range(self.batch_size)]
            for x_batch, y_batch, x_len_batch in _create_batches(
                    self.batch_size, x_test, y_test, x_test_lengths, pad=True, shuffle=False):
                if batch_i % 250 == 0:
                    log.info('Starting batch {}'.format(batch_i))
                feed_dict = {
                    self.input_placeholder: x_batch,
                    self.label_placeholder: y_batch,
                    self.len_placeholder: x_len_batch,
                    self.training_phase: 0
                }
                batch_predictions = self.session.run(self.softmax_output, feed_dict=feed_dict)
                if n_guesses is None:
                    n_guesses = batch_predictions.shape[1]

                #  Solution and explanation for column_index at
                #  http://stackoverflow.com/questions/33140674/argsort-for-a-multidimensional-ndarray
                ans_order = np.argsort(-batch_predictions, axis=1)
                sorted_labels = ans_order[:, :n_guesses]
                sorted_scores = batch_predictions[column_index, ans_order][:, :n_guesses]

                # We add an explicit np.copy so that we don't get a view into the original data.
                # The original data is much higher dimension and keeping a view over it is extremely
                # wasteful in terms of memory. It is better to pay the cost to copy a small portion
                # of it out.
                all_labels.append(np.copy(sorted_labels))
                all_scores.append(np.copy(sorted_scores))
                if batch_i % 250 == 0:
                    log.info('Finishing batch {}'.format(batch_i))
                batch_i += 1
            log.info('Done generating guesses, vstacking them...')

            return np.vstack(all_labels)[0:len(x_test)], np.vstack(all_scores)[0:len(x_test)]

    def save(self):
        self.saver.save(self.session, safe_path(DEEP_CNN_MODEL_TMP_PREFIX))

    def load(self):
        self.saver.restore(self.session, DEEP_CNN_MODEL_TMP_PREFIX)


DEFAULT_FIXED_PARAMS = dict(
    n_hidden_units=300, n_hidden_layers=2, word_dropout=.25, batch_size=128,
    learning_rate=.0005, max_epochs=100, nn_dropout=.25, max_patience=8
)


class CNNGuesser(AbstractGuesser):
    def __init__(self, cnn_params=DEFAULT_FIXED_PARAMS, use_wiki=False, min_answers=2):
        super().__init__()
        self.cnn_params = cnn_params
        self.model = None  # type: Union[None, CNNModel]
        self.embedding_lookup = None
        self.max_len = None  # type: Union[None, int]
        self.embeddings = None
        self.i_to_class = None
        self.class_to_i = None
        self.vocab = None
        self.n_classes = None
        self.use_wiki = use_wiki
        self.min_answers = min_answers

    @classmethod
    def targets(cls) -> List[str]:
        return [DEEP_CNN_PARAMS_TARGET]

    def qb_dataset(self):
        return QuizBowlDataset(self.min_answers)

    def train(self,
              training_data: TrainingData) -> None:
        log.info('Preprocessing training data...')
        x_train, y_train, _, x_test, y_test, _, vocab, class_to_i, i_to_class = preprocess_dataset(
            training_data)
        self.class_to_i = class_to_i
        self.i_to_class = i_to_class
        self.vocab = vocab

        if self.use_wiki:
            wiki_training_data = WikipediaDataset(self.min_answers).training_data()
            x_train_wiki, y_train_wiki, _, _, _, _, _, _, _ = preprocess_dataset(
                wiki_training_data, train_size=1, vocab=vocab, class_to_i=class_to_i,
                i_to_class=i_to_class)

        log.info('Creating embeddings...')
        embeddings, embedding_lookup = _load_embeddings(vocab=vocab)
        self.embeddings = embeddings
        self.embedding_lookup = self.embedding_lookup

        log.info('Converting dataset to embeddings...')
        x_train = [_convert_text_to_embeddings_indices(q, embedding_lookup) for q in x_train]
        x_train_lengths = _compute_lengths(x_train)

        x_test = [_convert_text_to_embeddings_indices(q, embedding_lookup) for q in x_test]
        x_test_lengths = _compute_lengths(x_test)

        if self.use_wiki:
            x_train_wiki = [_convert_text_to_embeddings_indices(q, embedding_lookup)
                            for q in x_train_wiki]
            x_train_lengths_wiki = _compute_lengths(x_train_wiki)
            x_train.extend(x_train_wiki)
            y_train.extend(y_train_wiki)
            x_train_lengths = np.concatenate([x_train_lengths, x_train_lengths_wiki])

        log.info('Computing number of classes and max paragraph length in words')
        self.n_classes = _compute_n_classes(training_data[1])
        # self.max_len = _compute_max_len(x_train)
        self.max_len = max([len(' '.join(sentences).split()) for sentences in training_data[0]])
        x_train = _tf_format(x_train, self.max_len, embeddings.shape[0])
        x_test = _tf_format(x_test, self.max_len, embeddings.shape[0])

        log.info('Training deep model...')
        self.model = CNNModel(self.cnn_params, self.max_len, self.n_classes)
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        train_losses, train_accuracies, holdout_losses, holdout_accuracies = self.model.train(
            x_train, y_train, x_train_lengths, x_test, y_test, x_test_lengths)

    def guess(self,
              questions: List[str], n_guesses: Optional[int]) -> List[List[Tuple[str, float]]]:
        log.info('Generating {} guesses for each of {} questions'.format(n_guesses, len(questions)))
        log.info('Converting text to embedding indices...')
        x_test = [_convert_text_to_embeddings_indices(
            tokenize_question(q), self.embedding_lookup) for q in questions]
        log.info('Computing question lengths...')
        x_test_lengths = _compute_lengths(x_test)
        log.info('Converting questions to tensorflow format...')
        x_test = _tf_format(x_test, self.max_len, self.embeddings.shape[0])
        x_test = np.array(x_test)
        self.model = CNNModel(self.cnn_params, self.max_len, self.n_classes)
        log.info('Starting Tensorflow model guessing...')
        guess_labels, guess_scores = self.model.guess(x_test, x_test_lengths, n_guesses)
        log.info('Guess generation and fetching top guesses done, converting to output format')
        all_guesses = []
        for i_row, score_row in zip(guess_labels, guess_scores):
            guesses = []
            for label, score in zip(i_row, score_row):
                guesses.append((self.i_to_class[label], score))
            all_guesses.append(guesses)
        return all_guesses

    def parameters(self):
        return {**self.cnn_params, 'use_wiki': self.use_wiki, 'min_answers': self.min_answers}

    @classmethod
    def load(cls, directory: str) -> AbstractGuesser:
        guesser = CNNGuesser()
        embeddings, embedding_lookup = _load_embeddings(root_directory=directory)
        guesser.embeddings = embeddings
        guesser.embedding_lookup = embedding_lookup
        params_path = os.path.join(directory, DEEP_CNN_PARAMS_TARGET)
        with open(params_path, 'rb') as f:
            params = pickle.load(f)
            guesser.max_len = params['max_len']
            guesser.class_to_i = params['class_to_i']
            guesser.i_to_class = params['i_to_class']
            guesser.vocab = params['vocab']
            guesser.n_classes = params['n_classes']
            if (guesser.max_len is None
                    or guesser.class_to_i is None
                    or guesser.i_to_class is None
                    or guesser.vocab is None
                    or guesser.n_classes is None):
                raise ValueError('Attempting to load uninitialized model parameters')
        model_path = os.path.join(directory, DEEP_CNN_MODEL_TARGET)
        shell('cp -r {} {}'.format(model_path, safe_path(DEEP_CNN_MODEL_TMP_DIR)))

        we_path = os.path.join(directory, TF_DAN_WE)
        shutil.copyfile(TF_DAN_WE_TMP, we_path)

        return guesser

    def save(self, directory: str) -> None:
        params_path = os.path.join(directory, DEEP_CNN_PARAMS_TARGET)
        with safe_open(params_path, 'wb') as f:
            if (self.max_len is None
                    or self.class_to_i is None
                    or self.i_to_class is None
                    or self.vocab is None
                    or self.n_classes is None):
                raise ValueError('Attempting to save uninitialized model parameters')
            pickle.dump({
                'max_len': self.max_len,
                'class_to_i': self.class_to_i,
                'i_to_class': self.i_to_class,
                'vocab': self.vocab,
                'n_classes': self.n_classes
            }, f)
        model_path = os.path.join(directory, DEEP_CNN_MODEL_TARGET)
        shell('cp -r {} {}'.format(DEEP_CNN_MODEL_TMP_DIR, safe_path(model_path)))
        we_path = os.path.join(directory, TF_DAN_WE)
        shutil.copyfile(TF_DAN_WE_TMP, safe_path(we_path))
