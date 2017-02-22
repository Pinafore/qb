import pickle
import time
import os
import shutil
from typing import Dict, List, Tuple, Union, Set

from qanta.guesser.abstract import AbstractGuesser
from qanta.datasets.abstract import AbstractDataset
from qanta.datasets.quiz_bowl import QuizBowlEvaluationDataset
from qanta.preprocess import preprocess_dataset, tokenize_question
from qanta.util.io import safe_open, safe_path, shell
from qanta import logging

import tensorflow as tf
import numpy as np

log = logging.get(__name__)
TF_DAN_WE_TMP = '/tmp/qanta/deep/tf_dan_we.pickle'
TF_DAN_WE = 'tf_dan_we.pickle'
GLOVE_WE = 'data/external/deep/glove.6B.300d.txt'
DEEP_DAN_MODEL_TMP_PREFIX = '/tmp/qanta/deep/tfdan'
DEEP_DAN_MODEL_TMP_DIR = '/tmp/qanta/deep'
DEEP_DAN_MODEL_TARGET = 'tfdan_dir'
DEEP_DAN_PARAMS_TARGET = 'dan_params.pickle'


def _make_layer(i: int, in_tensor, n_out, op,
                n_in=None, dropout_prob=None, batch_norm=False, batch_is_training=None):
    if batch_norm and batch_is_training is None:
        raise ValueError('if using batch norm then passing a training placeholder is required')
    w = tf.get_variable('W' + str(i), (in_tensor.get_shape()[1] if n_in is None else n_in, n_out),
                        dtype=tf.float32)
    if dropout_prob is not None:
        w = tf.nn.dropout(w, keep_prob=1 - dropout_prob)
    b = tf.get_variable('b' + str(i), n_out, dtype=tf.float32)
    out = tf.matmul(in_tensor, w) + b
    if batch_norm:
        out = tf.contrib.layers.batch_norm(
            out, center=True, scale=True, is_training=batch_is_training, scope='bn' + str(i))
    return (out if op is None else op(out)), w


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


def _load_embeddings(vocab=None):
    if os.path.exists(TF_DAN_WE_TMP):
        log.info('Loading word embeddings from cache')
        with safe_open(TF_DAN_WE_TMP, 'rb') as f:
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


class TFDanModel:
    def __init__(self, dan_params: Dict, max_len: int, n_classes: int):
        self.max_len = max_len
        self.n_classes = n_classes
        self.n_hidden_units = dan_params['n_hidden_units']
        self.n_hidden_layers = dan_params['n_hidden_layers']
        self.word_dropout = dan_params['word_dropout']
        self.nn_dropout = dan_params['nn_dropout']
        self.batch_size = dan_params['batch_size']
        self.init_scale = dan_params['init_scale']
        self.learning_rate = dan_params['learning_rate']
        self.max_epochs = dan_params['max_epochs']

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
        self.avg_embeddings = None
        self.word_dropout_var = None
        self.nn_dropout_var = None
        self.initial_embed = None
        self.mean_embeddings = None
        self.embed_and_zero = None

        # Set at runtime
        self.session = None

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
            self.embed_and_zero = tf.pad(self.initial_embed, [[0, 1], [0, 0]], mode='CONSTANT')
            self.input_placeholder = tf.placeholder(
                tf.int32, shape=(self.batch_size, self.max_len), name='input_placeholder')
            self.len_placeholder = tf.placeholder(
                tf.float32, shape=self.batch_size, name='len_placeholder')
            self.label_placeholder = tf.placeholder(
                tf.int32, shape=self.batch_size, name='label_placeholder')

            # (batch_size, max_len, embedding_dim)
            self.sent_vecs = tf.nn.embedding_lookup(self.embed_and_zero, self.input_placeholder)

            # Apply word level dropout
            self.word_dropout_var = tf.get_variable('word_dropout', (), dtype=tf.float32,
                                                    trainable=False)
            self.nn_dropout_var = tf.get_variable('nn_dropout', (), dtype=tf.float32,
                                                  trainable=False)
            drop_filter = tf.nn.dropout(
                tf.ones((self.max_len, 1)), keep_prob=1 - self.word_dropout_var)
            self.sent_vecs = self.sent_vecs * drop_filter
            in_dim = self.embed_and_zero.get_shape()[1]
            self.avg_embeddings = tf.reduce_sum(self.sent_vecs, 1) / tf.expand_dims(
                self.len_placeholder, 1)

            layer_out = self.avg_embeddings
            self.training_phase = tf.placeholder(tf.bool, name='phase')
            for i in range(self.n_hidden_layers):
                layer_out, w = _make_layer(
                    i, layer_out,
                    n_in=in_dim, n_out=self.n_hidden_units,
                    op=tf.nn.relu, dropout_prob=self.nn_dropout_var,
                    batch_norm=True, batch_is_training=self.training_phase
                )
                in_dim = None
            logits, w = _make_layer(self.n_hidden_layers, layer_out, n_out=self.n_classes, op=None)
            representation_layer = layer_out
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=tf.to_int64(self.label_placeholder))
            self.loss = tf.reduce_mean(self.loss)
            self.softmax_output = tf.nn.softmax(logits)
            preds = tf.to_int32(tf.argmax(logits, 1))
            self.batch_accuracy = tf.contrib.metrics.accuracy(preds, self.label_placeholder)
            accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(
                preds, self.label_placeholder)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                self.train_op = optimizer.minimize(self.loss)
            self.saver = tf.train.Saver()

    def train(self, x_train, y_train, x_train_lengths, x_test, y_test, x_test_lengths, save=True):
        with tf.Graph().as_default(), tf.Session() as session:
            self.build_tf_model()
            self.file_writer = tf.summary.FileWriter('output/tensorflow', session.graph)
            self.session = session
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
        self.session.run(tf.global_variables_initializer())

        max_accuracy = -1
        max_patience = 5
        patience = 5

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
                patience = max_patience
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
            fetches = self.loss, self.batch_accuracy, self.train_op
        else:
            fetches = self.loss, self.batch_accuracy

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

            accuracies.append(accuracy)
            losses.append(loss)
            batch_i += 1
        duration = time.time() - start_time
        return accuracies, losses, duration

    def guess(self, x_test, x_test_lengths):
        with tf.Graph().as_default(), tf.Session() as session:
            self.build_tf_model()
            self.session = session
            self.session.run(tf.global_variables_initializer())
            self.load()
            self.file_writer = tf.summary.FileWriter('output/tensorflow', session.graph)
            y_test = np.zeros((x_test.shape[0]))
            self.session.run(self.word_dropout_var.assign(0))
            self.session.run(self.nn_dropout_var.assign(0))
            predictions = []
            for x_batch, y_batch, x_len_batch in _create_batches(
                    self.batch_size, x_test, y_test, x_test_lengths, pad=True, shuffle=False):
                feed_dict = {
                    self.input_placeholder: x_batch,
                    self.label_placeholder: y_batch,
                    self.len_placeholder: x_len_batch,
                    self.training_phase: 0
                }
                batch_predictions = self.session.run(self.softmax_output, feed_dict=feed_dict)
                predictions.append(batch_predictions)
            return np.vstack(predictions)[:len(x_test)]

    def save(self):
        self.saver.save(self.session, safe_path(DEEP_DAN_MODEL_TMP_PREFIX))

    def load(self):
        self.saver.restore(self.session, DEEP_DAN_MODEL_TMP_PREFIX)


DEFAULT_DAN_PARAMS = dict(
    n_hidden_units=200, n_hidden_layers=2, word_dropout=.6, batch_size=256,
    learning_rate=.001, init_scale=.08, max_epochs=50, nn_dropout=0
)


class DANGuesser(AbstractGuesser):
    def __init__(self, dan_params=DEFAULT_DAN_PARAMS):
        super().__init__()
        self.dan_params = dan_params
        self.model = None  # type: Union[None, TFDanModel]
        self.embedding_lookup = None
        self.max_len = None  # type: Union[None, int]
        self.embeddings = None
        self.i_to_class = None
        self.class_to_i = None
        self.vocab = None
        self.n_classes = None

    @classmethod
    def targets(cls) -> List[str]:
        return [DEEP_DAN_PARAMS_TARGET]

    @property
    def requested_datasets(self) -> Dict[str, AbstractDataset]:
        return {
            'qb': QuizBowlEvaluationDataset()
        }

    def train(self,
              training_data: Dict[str, Tuple[List[List[str]], List[str]]]) -> None:
        qb_data = training_data['qb']

        log.info('Preprocessing training data...')
        x_train, y_train, x_test, y_test, vocab, class_to_i, i_to_class = preprocess_dataset(
            qb_data)
        self.class_to_i = class_to_i
        self.i_to_class = i_to_class
        self.vocab = vocab

        log.info('Creating embeddings...')
        embeddings, embedding_lookup = _load_embeddings(vocab=vocab)
        self.embeddings = embeddings
        self.embedding_lookup = self.embedding_lookup

        log.info('Converting dataset to embeddings...')
        x_train = [_convert_text_to_embeddings_indices(q, embedding_lookup) for q in x_train]
        x_train_lengths = _compute_lengths(x_train)

        x_test = [_convert_text_to_embeddings_indices(q, embedding_lookup) for q in x_test]
        x_test_lengths = _compute_lengths(x_test)

        log.info('Computing number of classes and max paragraph length in words')
        self.n_classes = _compute_n_classes(qb_data[1])
        self.max_len = _compute_max_len(x_train)
        x_train = _tf_format(x_train, self.max_len, embeddings.shape[0])
        x_test = _tf_format(x_test, self.max_len, embeddings.shape[0])

        log.info('Training deep model...')
        self.model = TFDanModel(self.dan_params, self.max_len, self.n_classes)
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        train_losses, train_accuracies, holdout_losses, holdout_accuracies = self.model.train(
            x_train, y_train, x_train_lengths, x_test, y_test, x_test_lengths)

    def guess(self, questions: List[str], n_guesses: int) -> List[List[Tuple[str, float]]]:
        log.info('Generating {} for each of {} questions'.format(n_guesses, len(questions)))
        x_test = [_convert_text_to_embeddings_indices(
            tokenize_question(q), self.embedding_lookup) for q in questions]
        x_test_lengths = _compute_lengths(x_test)
        x_test = _tf_format(x_test, self.max_len, self.embeddings.shape[0])
        x_test = np.array(x_test)
        self.model = TFDanModel(self.dan_params, self.max_len, self.n_classes)
        question_guesses = self.model.guess(x_test, x_test_lengths)
        ans_order = np.argsort(-question_guesses, axis=1)
        all_guesses = []
        for i_row, score_row in zip(ans_order, question_guesses):
            guesses = []
            for i, a_index in enumerate(i_row):
                if i < n_guesses:
                    guesses.append((self.i_to_class[a_index], score_row[a_index]))
                else:
                    break
            all_guesses.append(guesses)
        return all_guesses

    @classmethod
    def display_name(cls) -> str:
        return 'DAN'

    @classmethod
    def load(cls, directory: str) -> AbstractGuesser:
        guesser = DANGuesser()
        embeddings, embedding_lookup = _load_embeddings()
        guesser.embeddings = embeddings
        guesser.embedding_lookup = embedding_lookup
        params_path = os.path.join(directory, DEEP_DAN_PARAMS_TARGET)
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
        model_path = os.path.join(directory, DEEP_DAN_MODEL_TARGET)
        shell('cp -r {} {}'.format(model_path, DEEP_DAN_MODEL_TMP_DIR))

        we_path = os.path.join(directory, TF_DAN_WE)
        shutil.copyfile(TF_DAN_WE_TMP, we_path)

        return guesser

    def save(self, directory: str) -> None:
        params_path = os.path.join(directory, DEEP_DAN_PARAMS_TARGET)
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
        model_path = os.path.join(directory, DEEP_DAN_MODEL_TARGET)
        shell('cp -r {} {}'.format(DEEP_DAN_MODEL_TMP_DIR, safe_path(model_path)))
        we_path = os.path.join(directory, TF_DAN_WE)
        shutil.copyfile(TF_DAN_WE_TMP, safe_path(we_path))
