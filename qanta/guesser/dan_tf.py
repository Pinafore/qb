import pickle
from typing import Dict, List, Tuple
from qanta.guesser.abstract import AbstractGuesser
from qanta.datasets.abstract import AbstractDataset
from qanta.datasets.quiz_bowl import QuizBowlEvaluationDataset
from qanta.util.constants import DEEP_WE_TARGET

import tensorflow as tf


def _make_layer(i: int, in_tensor, n_out, op, n_in=None):
    W = tf.get_variable('W' + str(i), (in_tensor.get_shape()[1] if n_in is None else n_in, n_out),
                        dtype=tf.float32)
    b = tf.get_variable('b' + str(i), n_out, dtype=tf.float32)
    out = tf.matmul(in_tensor, W) + b
    return (out if op is None else op(out)), W


def _load_embeddings():
    with open(DEEP_WE_TARGET, 'rb') as f:
        embed = pickle.load(f)
        # For some reason embeddings are stored in column-major order
        embed = embed.T
    return embed


def _compute_n_classes(labels: List[str]):
    return len(set(labels))


def _compute_max_len(questions: List[List[str]]):
    max_len = 0
    for q in questions:
        length = sum(len(sentence) for sentence in q)
    return


class DANGuesser(AbstractGuesser):
    @classmethod
    def targets(cls) -> List[str]:
        return []

    def __init__(self, n_hidden_units=300, n_hidden_layers=2, word_dropout=.3, batch_size=128,
                 learning_rate=.0001, init_scale=.08):
        super().__init__()
        self.n_hidden_units = n_hidden_units
        self.n_hidden_layers = n_hidden_layers
        self.word_dropout = word_dropout
        self.batch_size = batch_size
        self.init_scale = init_scale
        self.learning_rate = learning_rate

    @property
    def requested_datasets(self) -> Dict[str, AbstractDataset]:
        return {
            'qb': QuizBowlEvaluationDataset()
        }

    def train(self,
              training_data: Dict[str, Tuple[List[List[str]], List[str]]]) -> None:
        with tf.Graph().as_default(), tf.Session as session:
            n_classes = _compute_n_classes(training_data['qb'][1])
            self._build_model(50, n_classes)

    def _preprocess_train(self, data: Tuple[List[List[str]], List[str]]):
        pass

    def _build_model(self, max_len, n_classes):
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
                tf.ones((max_len, 1)), keep_prob=1 - self.word_dropout)
            in_dim = embed_and_zero.get_shape()[1]
            layer_out = tf.reduce_sum(sent_vecs, 1) / tf.expand_dims(len_placeholder, 1)
            for i in range(self.n_hidden_layers):
                layer_out, w = _make_layer(i, layer_out, n_in=in_dim, n_out=self.n_hidden_units,
                                           op=tf.nn.relu)
                in_dim = None
            logits, w = _make_layer(self.n_hidden_layers, layer_out, n_out=n_classes, op=None)
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

