from itertools import repeat
import numpy as np
import pickle
from qanta import logging
from qanta.util.constants import DEEP_DAN_PARAMS_TARGET, DEEP_TRAIN_TARGET, DEEP_WE_TARGET
import tensorflow as tf
import time

log = logging.get(__name__)


def _make_layer(i, n_in, n_out):
    return (tf.get_variable('W' + str(i), (n_in, n_out), dtype=tf.float32),
            tf.get_variable('b' + str(i), n_out, dtype=tf.float32))


class TFDan:
    def __init__(self,
                 data_file,
                 batch_size=150,
                 n_epochs=61,
                 learning_rate=0.0001,
                 hidden_units=300,
                 n_layers=3,
                 is_train=True,
                 initial_embed=None):
        if not (is_train ^ (initial_embed is None)):
            raise ValueError('initial_embed should be None iff is_train is set')
        self._batch_size = batch_size
        self._is_train = is_train
        log.info('Loading data from {}'.format(data_file))
        self._load_data(data_file)
        log.info('Building model')
        self._build_model(hidden_units=hidden_units,
                          n_layers=n_layers,
                          learning_rate=learning_rate,
                          initial_embed=initial_embed)
        self._saver = tf.train.Saver()

    def _build_model(self, hidden_units, n_layers, learning_rate, initial_embed):
        if initial_embed is not None:
            self._embedding = tf.get_variable('embedding', initializer=tf.constant(initial_embed, dtype=tf.float32))
        else:
            self._embedding = tf.get_variable('embedding')

        zero_embed = tf.get_variable('zero_embed', initializer=tf.zeros((1, self._embedding.get_shape()[1])), trainable=False)

        embed_and_zero = tf.concat(0, (zero_embed, self._embedding))

        self._input_placeholder = tf.placeholder(tf.int32, shape=(None, self._max_len), name='input_placehodler')
        self._len_placeholder = tf.placeholder(tf.float32, shape=None, name='len_placeholder')
        self._label_placeholder = tf.placeholder(tf.int32, shape=(None), name='label_placeholder')

        # (batch_size, max_len, embedding_dim)
        sent_vecs = tf.nn.embedding_lookup(embed_and_zero, self._input_placeholder)
        # (batch_size, embedding_dim)
        mean_vecs = tf.reduce_sum(sent_vecs, 1) / tf.expand_dims(self._len_placeholder, 1)

        layers = [_make_layer(0, n_in=self._embedding.get_shape()[1], n_out=hidden_units)]
        for i in range(1, n_layers - 1):
            layers.append(_make_layer(i, n_in=hidden_units, n_out=hidden_units))

        layers.append(_make_layer(n_layers - 1, n_in=hidden_units, n_out=self._n_classes))

        layer_out = mean_vecs
        self._maxes = [tf.reduce_max(tf.abs(layer_out))]
        for W, b in layers[:-1]:
            layer_out = tf.nn.relu(tf.matmul(layer_out, W) + b)
            self._maxes.append(tf.reduce_max(tf.abs(layer_out)))
        final_W, final_b = layers[-1]
        logits = tf.matmul(layer_out, final_W) + final_b
        # logits = logits - tf.expand_dims(tf.reduce_max(logits, 1), 1)
        self._maxes.append(tf.reduce_min(tf.abs(logits)))

        self._loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, tf.to_int64(self._label_placeholder)))
        preds = tf.to_int32(tf.argmax(logits, 1))
        # correct_labels = tf.to_int32(tf.argmax(self._label_placeholder, 1))
        self._accuracy, self._accuracy_update = tf.contrib.metrics.streaming_accuracy(preds, self._label_placeholder)
        if not self._is_train:
            return

        optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)
        self._train_op = optimizer.minimize(self._loss)

    def _load_data(self, data_file, len_limit=400):
        """Handles data in format created by qanta/guesser/util/format_dan.py"""
        vecs = []
        labels = []
        max_len = 0
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
            for qs, l in data:
                q = []
                for i in range(len(qs)):
                    q.extend(qs[i])
                    if len(q) > len_limit:
                        break
                    max_len = max(len(q), max_len)
                    # Shift indices by 1 so that 0 can represent zero embedding
                    if len(q) > 0:
                        vecs.append([d + 1 for d in q])
                        labels.append(l[0])
            lens = []
            for v in vecs:
                # After end of question, pad with zero embedding
                lens.append(len(v))
                v.extend(repeat(0, max_len - len(v)))
        log.info('Loaded {} questions'.format(len(data)))
        log.info('{} total examples'.format(len(vecs)))
        log.info('Max example len: {}'.format(max_len))

        self._n_classes = max(labels) + 1
        self._labels = np.array(labels)

        self._data = np.array(vecs)
        self._lens = np.array(lens)
        self._labels = np.array(labels)
        self._max_len = max_len

    def _batches(self):
        order = list(range(len(self._data)))
        np.random.shuffle(order)
        for indices in (order[i:(i + self._batch_size)] for i in range(0, len(self._data), self._batch_size)):
            yield self._data[indices, :], self._lens[indices], self._labels[indices]

    def _run_epoch(self, session, epoch_num):
        # summary_writer = tf.train.SummaryWriter('/tmp/train', session.graph)
        total_loss = 0
        # Reset accuracy accumulators
        session.run(tf.initialize_local_variables())
        start_time = time.time()
        for i, (inputs, lens, labels) in enumerate(self._batches()):
            batch_start = time.time()
            fetches = ((self._loss, self._accuracy_update, self._train_op)
                       if self._is_train else
                       (self._loss, self._accuracy_update))
            feed_dict = {self._input_placeholder: inputs,
                         self._len_placeholder: lens,
                         self._label_placeholder: labels}

            # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            # run_metadata = tf.RunMetadata()
            # loss, *_ = session.run(fetches, feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
            loss, *_ = session.run(fetches, feed_dict=feed_dict)
            # summary_writer.add_run_metadata(run_metadata, 'step{}'.format(i))
            batch_duration = time.time() - batch_start
            log.info('Epoch: {} Batch: {} Loss: {} Duration: {}'.format(epoch_num, i, loss, batch_duration))
            total_loss += loss

        accuracy = session.run(self._accuracy)
        duration = time.time() - start_time
        log.info('Epoch: {} Avg loss: {} Accuracy: {}. Ran in {} seconds.'.format(epoch_num, total_loss / (i + 1), accuracy, duration))
        return accuracy

    def run(self, session, n_epochs):
        session.run(tf.initialize_all_variables())
        max_accuracy = 0
        for i in range(n_epochs):
            accuracy = self._run_epoch(session, i)
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                log.info('New best accuracy. Saving model')
                self._saver.save(session, DEEP_DAN_PARAMS_TARGET)


def train_dan(n_epochs):
    with open(DEEP_WE_TARGET, 'rb') as f:
        embed = pickle.load(f)
        # For some reason embeddings are stored in column-major order
        embed = embed.T

    with tf.Graph().as_default(), tf.Session() as session:
        scale = 0.08
        with tf.variable_scope('model',
                               reuse=None,
                               initializer=tf.random_uniform_initializer(minval=-scale, maxval=scale)):
            train_model = TFDan(data_file=DEEP_TRAIN_TARGET, is_train=True, initial_embed=embed)

        log.info('Training model')

        train_model.run(session, n_epochs)

if __name__ == '__main__':
    train_dan(50)
