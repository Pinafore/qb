import pickle
import time
import os
from typing import Dict, List, Tuple, Union, Set
from qanta.guesser.abstract import AbstractGuesser
from qanta.datasets.abstract import AbstractDataset
from qanta.datasets.quiz_bowl import QuizBowlEvaluationDataset
from qanta.preprocess import preprocess_dataset, tokenize_question
from qanta.util.constants import DEEP_DAN_MODEL_TARGET, DEEP_DAN_PARAMS_TARGET
from qanta.util.io import safe_open
from qanta import logging

import tensorflow as tf
from sklearn.cross_validation import train_test_split
import numpy as np

log = logging.get(__name__)
TF_DAN_WE = 'output/deep/tf_dan_we.pickle'
GLOVE_WE = 'data/external/deep/glove.6B.300d.txt'


def _make_layer(i: int, in_tensor, n_out, op, n_in=None):
    w = tf.get_variable('W' + str(i), (in_tensor.get_shape()[1] if n_in is None else n_in, n_out),
                        dtype=tf.float32)
    b = tf.get_variable('b' + str(i), n_out, dtype=tf.float32)
    out = tf.matmul(in_tensor, w) + b
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
        return np.array(embeddings), embedding_lookup


def _load_embeddings(vocab=None):
    if os.path.exists(TF_DAN_WE):
        log.info('Loading word embeddings from cache')
        with safe_open(TF_DAN_WE, 'rb') as f:
            return pickle.load(f)
    else:
        if vocab is None:
            raise ValueError('To create fresh embeddings a vocab is needed')
        with safe_open(TF_DAN_WE, 'wb') as f:
            log.info('Creating word embeddings and saving to cache')
            embed_and_lookup = _create_embeddings(vocab)
            pickle.dump(embed_and_lookup, f)
            return embed_and_lookup


def _convert_text_to_embeddings_indices(words: List[str], embedding_lookup: Dict[str, int]):
    w_indices = []
    for w in words:
        if w in embedding_lookup:
            w_indices.append(embedding_lookup[w])
    return w_indices


def _compute_n_classes(labels: List[str]):
    return len(set(labels))


def _compute_max_len(x_data: List[List[int]]):
    return max(len(x) for x in x_data)


def _tf_format(x_data: List[List[int]], max_len: int):
    """
    Pad with zeros if needed to max_len. Trim to max_len if needed. Convert all indices i to i + 1
    since the 0th index is the UNK otken
    :param x_data:
    :param max_len:
    :return:
    """
    for row in x_data:
        for i in range(len(row)):
            row[i] += 1
    for i in range(len(x_data)):
        row = x_data[i]
        while len(row) < max_len:
            row.append(0)
        x_data[i] = x_data[i][:max_len]
    return x_data


def _create_batches(batch_size, x_data: np.ndarray, y_data: np.ndarray, x_lengths: np.ndarray):
    if type(x_data) != np.ndarray or type(y_data) != np.ndarray:
        raise ValueError('x and y must be numpy arrays')
    if len(x_data) != len(y_data):
        raise ValueError('x and y must have the same dimension')
    n = len(x_data)
    order = list(range(n))
    np.random.shuffle(order)
    for i in range(0, n, batch_size):
        if len(order[i:i + batch_size]) == batch_size:
            yield (
                x_data[order[i:i + batch_size]],
                y_data[order[i:i + batch_size]],
                x_lengths[order[i:i + batch_size]]
            )
        else:
            break


def _compute_lengths(x_data):
    return np.array([len(x) for x in x_data])


class TFDanModel:
    def __init__(self, dan_params: Dict, max_len: int, n_classes: int):
        self.max_len = max_len
        self.n_classes = n_classes
        self.n_hidden_units = dan_params['n_hidden_units']
        self.n_hidden_layers = dan_params['n_hidden_layers']
        self.word_dropout = dan_params['word_dropout']
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

        # Set at runtime
        self.session = None

    def build_tf_model(self):
        with tf.variable_scope('dan', reuse=None, initializer=tf.random_uniform_initializer(
                minval=-self.init_scale, maxval=self.init_scale)):
            embedding, embedding_word_lookup = _load_embeddings()
            initial_embed = tf.get_variable(
                'embedding',
                initializer=tf.constant(embedding, dtype=tf.float32)
            )
            embed_and_zero = tf.pad(initial_embed, [[1, 0], [0, 0]], mode='CONSTANT')
            self.input_placeholder = tf.placeholder(
                tf.int32, shape=(self.batch_size, self.max_len), name='input_placeholder')
            self.len_placeholder = tf.placeholder(
                tf.float32, shape=self.batch_size, name='len_placeholder')
            self.label_placeholder = tf.placeholder(
                tf.int32, shape=self.batch_size, name='label_placeholder')

            # (batch_size, max_len, embedding_dim)
            self.sent_vecs = tf.nn.embedding_lookup(embed_and_zero, self.input_placeholder)

            # Apply word level dropout
            drop_filter = tf.nn.dropout(
                tf.ones((self.max_len, 1)), keep_prob=1 - self.word_dropout)
            in_dim = embed_and_zero.get_shape()[1]
            layer_out = tf.reduce_sum(self.sent_vecs, 1) / tf.expand_dims(self.len_placeholder, 1)
            for i in range(self.n_hidden_layers):
                layer_out, w = _make_layer(i, layer_out, n_in=in_dim, n_out=self.n_hidden_units,
                                           op=tf.nn.relu)
                in_dim = None
            logits, w = _make_layer(self.n_hidden_layers, layer_out, n_out=self.n_classes, op=None)
            representation_layer = layer_out
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits, tf.to_int64(self.label_placeholder))
            self.loss = tf.reduce_mean(self.loss)
            self.softmax_output = tf.nn.softmax(logits)
            preds = tf.to_int32(tf.argmax(logits, 1))
            self.batch_accuracy = tf.contrib.metrics.accuracy(preds, self.label_placeholder)
            accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(
                preds, self.label_placeholder)
            optimizer = tf.train.AdamOptimizer()
            self.train_op = optimizer.minimize(self.loss)
            self.saver = tf.train.Saver()

    def train(self, x_data, y_data, x_lengths):
        with tf.Graph().as_default(), tf.Session() as session:
            self.build_tf_model()
            self.file_writer = tf.summary.FileWriter('output/tensorflow', session.graph)
            self.session = session
            train_losses, train_accuracies, holdout_losses, holdout_accuracies = self._train(
                x_data, y_data, x_lengths, self.max_epochs)
            return train_losses, train_accuracies, holdout_losses, holdout_accuracies

    def _train(self, x_data, y_data, x_lengths, n_epochs: int):
        self.session.run(tf.global_variables_initializer())

        max_accuracy = -1
        max_patience = 4
        patience = 4

        train_accuracies = []
        train_losses = []

        holdout_accuracies = []
        holdout_losses = []

        x_train, x_test, y_train, y_test, x_lengths_train, x_lengths_test = train_test_split(
            x_data, y_data, x_lengths, train_size=.90)

        for i in range(n_epochs):
            # Training Epoch
            accuracies, losses, duration = self.run_epoch(
                x_train, y_train, x_lengths_train)
            log.info(
                'Train Epoch: {} Avg loss: {:.4f} Accuracy: {:.4f}. Ran in {:.4f} seconds.'.format(
                    i, np.average(losses), np.average(accuracies), duration))
            train_accuracies.append(accuracies)
            train_losses.append(losses)

            # Validation Epoch
            val_accuracies, val_losses, val_duration = self.run_epoch(
                x_test, y_test, x_lengths_test, train=False)
            val_accuracy = np.average(val_accuracies)
            log.info(
                'Val Epoch: {} Avg loss: {:.4f} Accuracy: {:.4f}. Ran in {:.4f} seconds.'.format(
                    i, np.average(val_losses), val_accuracy, val_duration))
            holdout_accuracies.append(val_accuracies)
            holdout_losses.append(val_losses)

            # Early Stopping and Model Saving
            patience -= 1
            if val_accuracy > max_accuracy:
                max_accuracy = val_accuracy
                log.info('New best accuracy, saving model')
                patience = max_patience
                self.saver.save(self.session, DEEP_DAN_MODEL_TARGET)

            if patience == 0:
                break

        return train_losses, train_accuracies, holdout_losses, holdout_accuracies

    def run_epoch(self, x_data, y_data, x_lengths, train=True):
        total_loss = 0
        start_time = time.time()
        accuracies = []
        losses = []
        if train:
            fetches = self.loss, self.batch_accuracy, self.train_op
        else:
            fetches = self.loss, self.batch_accuracy

        batch_i = 0
        for x_batch, y_batch, x_len_batch in _create_batches(
                self.batch_size, x_data, y_data, x_lengths):
            feed_dict = {
                self.input_placeholder: x_batch,
                self.len_placeholder: x_len_batch,
                self.label_placeholder: y_batch
            }
            returned = self.session.run(fetches, feed_dict=feed_dict)
            loss = returned[0]
            accuracy = returned[1]

            accuracies.append(accuracy)
            total_loss += loss
            losses.append(loss)
            batch_i += 1
        duration = time.time() - start_time
        return accuracies, losses, duration

    def test(self, x_test):
        fetches = (self.softmax_output,)

    def save(self):
        self.saver.save(self.session, DEEP_DAN_MODEL_TARGET)

    def load(self):
        pass

DEFAULT_DAN_PARAMS = dict(
    n_hidden_units=200, n_hidden_layers=2, word_dropout=.3, batch_size=128,
    learning_rate=.0001, init_scale=.08, max_epochs=50
)


class DANGuesser(AbstractGuesser):
    def __init__(self, dan_params=DEFAULT_DAN_PARAMS):
        super().__init__()
        self.dan_params = dan_params
        self.model = None  # type: Union[None, TFDanModel]
        self.embedding_lookup = None
        self.max_len = None  # type: Union[None, int]

    @classmethod
    def targets(cls) -> List[str]:
        return []

    @property
    def requested_datasets(self) -> Dict[str, AbstractDataset]:
        return {
            'qb': QuizBowlEvaluationDataset()
        }

    def train(self,
              training_data: Dict[str, Tuple[List[List[str]], List[str]]]) -> None:
        qb_data = training_data['qb']

        log.info('Preprocessing training data...')
        x_train, y_train, vocab, class_to_i, i_to_class = preprocess_dataset(qb_data)

        log.info('Creating embeddings...')
        _, embedding_lookup = _load_embeddings(vocab=vocab)
        self.embedding_lookup = self.embedding_lookup

        log.info('Converting dataset to embeddings...')
        x_train = [_convert_text_to_embeddings_indices(q, embedding_lookup) for q in x_train]
        x_lengths = _compute_lengths(x_train)

        log.info('Computing number of classes and max paragraph length in words')
        n_classes = _compute_n_classes(qb_data[1])
        self.max_len = _compute_max_len(x_train)
        x_train = _tf_format(x_train, self.max_len)

        log.info('Training deep model...')
        self.model = TFDanModel(self.dan_params, self.max_len, n_classes)
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        train_losses, train_accuracies, holdout_losses, holdout_accuracies = self.model.train(
            x_train, y_train, x_lengths)

    def load(self, directory: str) -> None:
        self.model.load()
        _, embedding_lookup = _load_embeddings()
        self.embedding_lookup = embedding_lookup
        with open(DEEP_DAN_PARAMS_TARGET, 'rb') as f:
            params = pickle.load(f)
            self.max_len = params['max_len']

    def guess(self, questions: List[str], n_guesses: int) -> List[List[Tuple[str, float]]]:
        x_test = [_convert_text_to_embeddings_indices(
            tokenize_question(q), self.embedding_lookup) for q in questions]
        x_test = _tf_format(x_test, self.max_len)
        x_test = np.array(x_test)
        y_test = self.model.test(x_test)

    @property
    def display_name(self) -> str:
        return 'DAN'

    def save(self, directory: str) -> None:
        self.model.save()
        with safe_open(DEEP_DAN_PARAMS_TARGET, 'wb') as f:
            pickle.dump({'max_len': self.max_len}, f)

    def score(self, question: str, guesses: List[str]) -> List[float]:
        pass

