import pickle
import os
from typing import Dict, List, Tuple
from qanta.guesser.abstract import AbstractGuesser
from qanta.datasets.abstract import AbstractDataset
from qanta.datasets.quiz_bowl import QuizBowlEvaluationDataset
from qanta.preprocess import preprocess_dataset, create_embeddings
from qanta import logging

import tensorflow as tf

log = logging.get(__name__)
TF_DAN_WE = 'output/deep/tf_dan_we.pickle'


def _make_layer(i: int, in_tensor, n_out, op, n_in=None):
    W = tf.get_variable('W' + str(i), (in_tensor.get_shape()[1] if n_in is None else n_in, n_out),
                        dtype=tf.float32)
    b = tf.get_variable('b' + str(i), n_out, dtype=tf.float32)
    out = tf.matmul(in_tensor, W) + b
    return (out if op is None else op(out)), W


def _load_embeddings(vocab=None):
    if os.path.exists(TF_DAN_WE):
        log.info('Loading word embeddings from cache')
        with open(TF_DAN_WE, 'rb') as f:
            return pickle.load(f)
    else:
        if vocab is None:
            raise ValueError('To create fresh embeddings a vocab is needed')
        with open(TF_DAN_WE, 'wb') as f:
            log.info('Creating word embeddings and saving to cache')
            embed_and_lookup = create_embeddings(vocab)
            pickle.dump(embed_and_lookup, f)
            return embed_and_lookup


def _convert_text_to_embeddings(text: List[str], embeddings, embedding_lookup):
    w_indices = []
    for w in text:
        if w in embedding_lookup:
            w_indices.append(embeddings[embedding_lookup[w]])
    return w_indices


def _compute_n_classes(labels: List[str]):
    return len(set(labels))


def _compute_max_len(x_data: List[List[int]]):
    return max(len(x) for x in x_data)


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

    def build_tf_model(self):
        with tf.variable_scope('dan', reuse=None, initializer=tf.random_uniform_initializer(
                minval=-self.init_scale, maxval=self.init_scale)):
            initial_embed = tf.get_variable(
                'embedding',
                initializer=tf.constant(_load_embeddings(), dtype=tf.float32)
            )
            embed_and_zero = tf.pad(initial_embed, [[1, 0], [0, 0]], mode='CONSTANT')
            input_placeholder = tf.placeholder(
                tf.int32, shape=(self.batch_size, None), name='input_placeholder')
            len_placeholder = tf.placeholder(
                tf.float32, shape=self.batch_size, name='len_placeholder')
            label_placeholder = tf.placeholder(
                tf.int32, shape=self.batch_size, name='label_placeholder')
            # (batch_size, max_len, embedding_dim)
            sent_vecs = tf.nn.embedding_lookup(embed_and_zero, input_placeholder)

            # Apply word level dropout
            drop_filter = tf.nn.dropout(
                tf.ones((self.max_len, 1)), keep_prob=1 - self.word_dropout)
            in_dim = embed_and_zero.get_shape()[1]
            layer_out = tf.reduce_sum(sent_vecs, 1) / tf.expand_dims(len_placeholder, 1)
            for i in range(self.n_hidden_layers):
                layer_out, w = _make_layer(i, layer_out, n_in=in_dim, n_out=self.n_hidden_units,
                                           op=tf.nn.relu)
                in_dim = None
            logits, w = _make_layer(self.n_hidden_layers, layer_out, n_out=self.n_classes, op=None)
            representation_layer = layer_out
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits, tf.to_int64(label_placeholder))
            loss = tf.reduce_mean(loss)
            softmax_output = tf.nn.softmax(logits)
            preds = tf.to_int32(tf.argmax(logits, 1))
            batch_accuracy = tf.contrib.metrics.accuracy(preds, label_placeholder)
            accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(
                preds, label_placeholder)
            optimizer = tf.train.AdamOptimizer()
            train_op = optimizer.minimize(loss)

    def train(self, x_train, y_train):
        with tf.Graph().as_default(), tf.Session() as session:
            self.build_tf_model()
            self._train(session, self.max_epochs)

    def _train(self, session: tf.Session, n_epochs: int):
        session.run(tf.initialize_all_variables())

        train_accuracies = []
        train_losses = []

        holdout_accuracies = []
        holdout_losses = []

DEFAULT_DAN_PARAMS = dict(
    n_hidden_units=200, n_hidden_layers=2, word_dropout=.3, batch_size=128,
                 learning_rate=.0001, init_scale=.08, max_epochs=50
)


class DANGuesser(AbstractGuesser):
    @classmethod
    def targets(cls) -> List[str]:
        return []

    def __init__(self, dan_params=DEFAULT_DAN_PARAMS):
        super().__init__()
        self.dan_params = dan_params

    @property
    def requested_datasets(self) -> Dict[str, AbstractDataset]:
        return {
            'qb': QuizBowlEvaluationDataset()
        }

    def train(self,
              training_data: Dict[str, Tuple[List[List[str]], List[str]]]) -> None:
        qb_data = training_data['qb']
        x_train, y_train, vocab = preprocess_dataset(qb_data)
        embeddings, embedding_lookup = _load_embeddings(vocab=vocab)
        x_train = [_convert_text_to_embeddings(q, embeddings, embedding_lookup) for q in x_train]

        n_classes = _compute_n_classes(qb_data[1])
        max_len = _compute_max_len(x_train)
        model = TFDanModel(self.dan_params, max_len, n_classes)
        model.train(x_train, y_train)

    def load(self, directory: str) -> None:
        pass

    def guess(self, questions: List[str], n_guesses: int) -> List[
        List[Tuple[str, float]]]:
        pass

    @property
    def display_name(self) -> str:
        return 'DAN'

    def save(self, directory: str) -> None:
        pass

    def score(self, question: str, guesses: List[str]) -> List[float]:
        pass

