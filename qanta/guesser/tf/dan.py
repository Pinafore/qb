import heapq
from itertools import repeat
import numpy as np
import pickle
from qanta import logging
from qanta.util.constants import (DEEP_DAN_PARAMS_TARGET, DEEP_TF_PARAMS_TARGET, DEEP_TRAIN_TARGET,
                                  DEEP_DEV_TARGET, DEEP_WE_TARGET, EVAL_RES_TARGET,
                                  N_GUESSES)
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
                 initial_embed=None,
                 embedding_shape=None,
                 n_classes=None,
                 word_drop=0.3,
                 rho=10**-5):
        if is_train and (initial_embed is None or n_classes is not None or embedding_shape is not None):
            raise ValueError('Illegal values for training model')
        elif not is_train and (initial_embed is not None or n_classes is None or embedding_shape is None):
            raise ValueError('Illegal values for non-training model')

        self._batch_size = batch_size
        self._is_train = is_train
        log.info('Loading data from {}'.format(data_file))
        self._load_data(data_file)
        if not is_train:
            self._n_classes = n_classes

        log.info('Got {} classes'.format(self._n_classes))

        log.info('Building model')
        self._build_model(hidden_units=hidden_units,
                          n_layers=n_layers,
                          learning_rate=learning_rate,
                          initial_embed=initial_embed,
                          word_drop=word_drop,
                          rho=rho,
                          embedding_shape=embedding_shape)
        self._saver = tf.train.Saver()

    def _build_model(self, hidden_units, n_layers, learning_rate, word_drop, rho, initial_embed, embedding_shape):
        if initial_embed is not None:
            self._embedding = tf.get_variable('embedding', initializer=tf.constant(initial_embed, dtype=tf.float32))
        else:
            self._embedding = tf.get_variable('embedding', shape=embedding_shape, dtype=tf.float32)

        zero_embed = tf.get_variable('zero_embed', initializer=tf.zeros((1, self._embedding.get_shape()[1])), trainable=False)

        embed_and_zero = tf.concat(0, (zero_embed, self._embedding))

        self._input_placeholder = tf.placeholder(tf.int32, shape=(None, None), name='input_placeholder')
        self._len_placeholder = tf.placeholder(tf.float32, shape=None, name='len_placeholder')
        self._label_placeholder = tf.placeholder(tf.int32, shape=None, name='label_placeholder')

        # (batch_size, max_len, embedding_dim)
        sent_vecs = tf.nn.embedding_lookup(embed_and_zero, self._input_placeholder)

        # Apply dropout at word level
        if self._is_train:
            drop_filter = tf.nn.dropout(tf.ones((self._max_len, 1)), keep_prob=(1 - word_drop))
            sent_vecs = sent_vecs * drop_filter

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

        self._loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, tf.to_int64(self._label_placeholder)))
        # for W, b in layers:
        #     self._loss += tf.nn.l2_loss(W) * rho
        # Used for labeling
        self._softmax_output = tf.nn.softmax(logits)

        preds = tf.to_int32(tf.argmax(logits, 1))
        # correct_labels = tf.to_int32(tf.argmax(self._label_placeholder, 1))
        self._accuracy, self._accuracy_update = tf.contrib.metrics.streaming_accuracy(preds, self._label_placeholder)
        if not self._is_train:
            return

        # optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)
        optimizer = tf.train.AdamOptimizer()
        self._train_op = optimizer.minimize(self._loss)

    def _load_data(self, data_file, len_limit=400):
        """Handles data in format created by qanta/guesser/util/format_dan.py"""
        vecs = []
        labels = []
        complete = []
        max_len = 0
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
            for qs, l in data:
                q = []
                for i, sent in enumerate(qs):
                    q.extend(qs[i])
                    if self._is_train and len(q) > len_limit:
                        break
                    max_len = max(len(q), max_len)
                    if len(q) > 0:
                        # Shift indices by 1 so that 0 can represent zero embedding
                        vecs.append([d + 1 for d in q])
                        labels.append(l[0])
                    complete.append(i == len(qs) - 1)
            lens = []
            for v in vecs:
                lens.append(len(v))
                if self._is_train:
                    # After end of question, pad with zero embedding
                    v.extend(repeat(0, max_len - len(v)))
        log.info('Loaded {} questions'.format(len(data)))
        log.info('{} total examples'.format(len(vecs)))
        log.info('Max example len: {}'.format(max_len))

        # Only need to get number of classes if building a model from scratch
        self._n_classes = max(labels) + 1
        self._labels = np.array(labels)

        # Conversion of each v to array matters if data is jagged (which it will be for non-training models)
        self._data = np.array([np.array(v) for v in vecs])
        self._lens = np.array(lens)
        self._complete = complete
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
            total_loss += loss
            batch_duration = time.time() - batch_start
            log.info('Epoch: {} Batch: {} Loss: {} Duration: {}'.format(epoch_num, i, loss, batch_duration))

        accuracy = session.run(self._accuracy)
        duration = time.time() - start_time
        log.info('Epoch: {} Avg loss: {} Accuracy: {}. Ran in {} seconds.'.format(epoch_num, total_loss / (i + 1), accuracy, duration))
        return accuracy

    def train(self, session, n_epochs):
        if not self._is_train:
            raise ValueError('To use a non-train model, call label() instead')
        session.run(tf.initialize_all_variables())
        max_accuracy = -1
        for i in range(n_epochs):
            accuracy = self._run_epoch(session, i)
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                log.info('New best accuracy. Saving model')
                self._saver.save(session, DEEP_DAN_PARAMS_TARGET)
                with open(DEEP_TF_PARAMS_TARGET, 'wb') as f:
                    save_vals = {'n_classes': self._n_classes, 'embedding_shape': self._embedding.get_shape()}
                    pickle.dump(save_vals, f)

    def evaluate(self, session):
        """Generate softmax output for all examples in dataset"""
        self._saver.restore(session, DEEP_DAN_PARAMS_TARGET)
        session.run(tf.initialize_local_variables())
        results = []
        count = 0
        for i, (in_array, lens, labels, complete) in enumerate(zip(self._data, self._lens, self._labels, self._complete)):
            if not complete:
                continue
            fetches = (self._softmax_output, self._accuracy_update)
            feed_dict = {self._input_placeholder: in_array, self._len_placeholder: lens, self._label_placeholder: labels}
            feed_dict = {k: np.expand_dims(v, 0) for k, v in feed_dict.items()}
            softmax_output, _ = session.run(fetches, feed_dict=feed_dict)
            results.append((i, np.squeeze(softmax_output)))
            count += 1
            if count % 1000 == 0:
                log.info('Labeled {} examples'.format(count))
        accuracy = session.run(self._accuracy)
        recalls = self._recall_at_n(results)
        return accuracy, recalls

    def _recall_at_n(self, probs, n_max=N_GUESSES):
        """Compute recall@N for all N up to n_max"""
        # Get indices of all examples which are full questions
        num_correct = 0
        ordered_probs = [(i, heapq.nlargest(n_max, range(self._n_classes), key=p.__getitem__)) for i, p in probs]
        total = len(ordered_probs)
        incorrect = {i: j for i, (j, _) in enumerate(ordered_probs)}
        result = []
        for i in range(0, n_max):
            log.info('Computing recall@{}'.format(i))
            to_remove = []
            for ex_num, label_index in incorrect.items():
                if ordered_probs[ex_num][1][i] == self._labels[label_index]:
                    to_remove.append(ex_num)
                    num_correct += 1
            for r in to_remove:
                del incorrect[r]

            result.append(num_correct / total)
        return result


def train_dan(n_epochs):
    with open(DEEP_WE_TARGET, 'rb') as f:
        embed = pickle.load(f)
        # For some reason embeddings are stored in column-major order
        embed = embed.T

    with tf.Graph().as_default(), tf.Session() as session:
        scale = 0.08
        with tf.variable_scope('dan',
                               reuse=None,
                               initializer=tf.random_uniform_initializer(minval=-scale, maxval=scale)):
            train_model = TFDan(data_file=DEEP_TRAIN_TARGET, is_train=True, initial_embed=embed)
        log.info('Training model')

        train_model.train(session, n_epochs)


def evaluate():
    with open(DEEP_TF_PARAMS_TARGET, 'rb') as f:
        extra_params = pickle.load(f)
        n_classes = extra_params['n_classes']
        embedding_shape = extra_params['embedding_shape']
    with tf.Graph().as_default(), tf.Session() as session:
        with tf.variable_scope('dan', reuse=None):
            dev_model = TFDan(
                data_file=DEEP_DEV_TARGET,
                is_train=False,
                initial_embed=None,
                n_classes=n_classes,
                embedding_shape=embedding_shape)
        log.info('Evaluating model on dev')
        dev_accuracy, dev_recalls = dev_model.evaluate(session)
        log.info('Accuracy on dev: {}'.format(dev_accuracy))
        with open(EVAL_RES_TARGET, 'wb') as f:
            pickle.dump(dev_recalls, f)

if __name__ == '__main__':
    train_dan(50)
