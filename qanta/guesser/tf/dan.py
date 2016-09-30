from collections import defaultdict
import heapq
from itertools import repeat
import numpy as np
import pickle
from qanta import logging
from qanta.util.constants import (DEEP_DAN_PARAMS_TARGET, DEEP_TF_PARAMS_TARGET, DEEP_TRAIN_TARGET,
                                  DEEP_DEV_TARGET, DEEP_WE_TARGET, DEEP_WIKI_TARGET,
                                  EVAL_RES_TARGET, N_GUESSES, REPRESENTATION_RES_TARGET)
import random
import tensorflow as tf
import time

log = logging.get(__name__)


def _make_layer(i, in_tensor, n_out, op, n_in=None):
    W = tf.get_variable('W' + str(i), (in_tensor.get_shape()[1] if n_in is None else n_in, n_out), dtype=tf.float32)
    b = tf.get_variable('b' + str(i), n_out, dtype=tf.float32)
    out = tf.matmul(in_tensor, W) + b
    return (out if op is None else op(out)), W


class TFDan:
    def __init__(self,
                 data_files,
                 batch_size=128,
                 n_epochs=61,
                 learning_rate=0.0001,
                 hidden_units=300,
                 n_prediction_layers=2,
                 is_train=True,
                 adversarial=False,
                 domain_classifier_weight=0.02,
                 adversarial_interval=3,
                 n_representation_layers=2,
                 lstm_representation=True,
                 initial_embed=None,
                 embedding_shape=None,
                 label_map=None,
                 n_classes=None,
                 word_drop=0.3,
                 lstm_dropout_prob=0.5,
                 rho=10**-5,
                 use_weights=False):
        if is_train and (initial_embed is None or n_classes is not None or embedding_shape is not None):
            raise ValueError('Illegal values for training model')
        elif not is_train and (initial_embed is not None or n_classes is None or embedding_shape is None):
            raise ValueError('Illegal values for non-training model')

        self._batch_size = batch_size if is_train else 1
        self._is_train = is_train
        self._lstm_representation = lstm_representation
        self._adversarial = adversarial
        self._adversarial_interval = adversarial_interval
        self._use_weights = use_weights
        self._load_data(data_files, label_map=label_map)
        if not is_train:
            self._n_classes = n_classes

        log.info('Got {} classes'.format(self._n_classes))

        log.info('Building model')
        self._build_model(hidden_units=hidden_units,
                          n_prediction_layers=n_prediction_layers,
                          domain_classifier_weight=domain_classifier_weight,
                          n_representation_layers=n_representation_layers,
                          learning_rate=learning_rate,
                          initial_embed=initial_embed,
                          word_drop=word_drop,
                          lstm_dropout_prob=lstm_dropout_prob,
                          rho=rho,
                          embedding_shape=embedding_shape)
        self._saver = tf.train.Saver()

    def _build_model(
            self,
            hidden_units,
            n_prediction_layers,
            domain_classifier_weight,
            n_representation_layers,
            learning_rate,
            word_drop,
            lstm_dropout_prob,
            rho,
            initial_embed,
            embedding_shape,
            representation_depth=2):
        if initial_embed is not None:
            self._embedding = tf.get_variable('embedding', initializer=tf.constant(initial_embed, dtype=tf.float32))
        else:
            self._embedding = tf.get_variable('embedding', shape=embedding_shape, dtype=tf.float32)

        embed_and_zero = tf.pad(self._embedding, [[1, 0], [0, 0]], mode='CONSTANT')

        batch_dim = self._batch_size if self._is_train else 1
        self._input_placeholder = tf.placeholder(tf.int32, shape=(batch_dim, None), name='input_placeholder')
        self._len_placeholder = tf.placeholder(tf.float32, shape=batch_dim, name='len_placeholder')
        self._label_placeholder = tf.placeholder(tf.int32, shape=batch_dim, name='label_placeholder')
        if self._use_weights:
            self._weight_placeholder = tf.placeholder(tf.float32, shape=None, name='weight_placeholder')
        if self._adversarial:
            self._domain_gate_placeholder = tf.placeholder(tf.float32, shape=(), name='domain_gate_placeholder')
            self._domain_placeholder = tf.placeholder(tf.float32, shape=None, name='domain_placeholder')

        # (batch_size, max_len, embedding_dim)
        sent_vecs = tf.nn.embedding_lookup(embed_and_zero, self._input_placeholder)

        # Apply dropout at word level
        if self._is_train:
            drop_filter = tf.nn.dropout(tf.ones((self._max_len, 1)), keep_prob=(1 - word_drop))
            sent_vecs = sent_vecs * drop_filter

        # Store layer weights for use in regularization
        weights = []

        # (batch_size, embedding_dim) mean of embeddings
        in_dim = embed_and_zero.get_shape()[1]
        with tf.variable_scope('representation_net'):
            if self._lstm_representation:
                self._representation_layer = self._build_lstm(
                        input_layer=sent_vecs,
                        hidden_units=hidden_units,
                        lengths=self._len_placeholder,
                        dropout_prob=lstm_dropout_prob,
                        n_layers=n_representation_layers)
            else:
                layer_out = tf.reduce_sum(sent_vecs, 1) / tf.expand_dims(self._len_placeholder, 1)
                for i in range(n_representation_layers):
                    layer_out, w = _make_layer(i, layer_out, n_in=in_dim, n_out=hidden_units, op=tf.nn.relu)
                    in_dim = None
                self._representation_layer = layer_out
        in_dim = hidden_units

        with tf.variable_scope('prediction_net'):
            layer_out = self._representation_layer
            for i in range(n_prediction_layers - 1):
                layer_out, w = _make_layer(i, layer_out, n_in=in_dim, n_out=hidden_units, op=tf.nn.relu)
                weights.append(w)
                in_dim = None

            logits, w = _make_layer(n_prediction_layers - 1, layer_out, n_out=self._n_classes, op=None)
            weights.append(w)
            # logits = logits - tf.expand_dims(tf.reduce_max(logits, 1), 1)

            self._loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, tf.to_int64(self._label_placeholder))
        if self._use_weights:
            self._loss *= self._weight_placeholder
            self._loss = tf.reduce_sum(self._loss) / tf.reduce_sum(self._weight_placeholder)
        else:
            self._loss = tf.reduce_mean(self._loss)

        if self._adversarial:
            domain_loss, layers = self._build_domain_classifier(self._representation_layer, hidden_units, domain_classifier_weight)
            weights.extend(layers)
            # Downeighting of domain loss happens in modified gradient
            self._loss += domain_loss

        # for W in weights:
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

    def _build_domain_classifier(self, input_layer, hidden_units, domain_classifier_weight, n_layers=2):
        @tf.RegisterGradient('GradientReversal')
        def gradient_reversal(op, grads):
            return -grads * domain_classifier_weight * self._domain_gate_placeholder

        with tf.variable_scope('domain_classifier'):
            with tf.get_default_graph().gradient_override_map({'Identity': 'GradientReversal'}):
                # batch_size, num_units
                reversal_layer = tf.identity(input_layer)
            layer_out = reversal_layer
            weights = []
            for i in range(n_layers - 1):
                layer_out, w = _make_layer(i, layer_out, n_out=hidden_units, op=tf.nn.relu)
                weights.append(w)

            logits, w = _make_layer(n_layers - 1, layer_out, n_out=1, op=None)
            weights.append(w)
            domain_preds = tf.to_int32(tf.sign(logits))

            self._domain_accuracy, self._domain_accuracy_update = tf.contrib.metrics.streaming_accuracy(
                tf.squeeze(domain_preds), 2 * tf.to_int32(self._domain_placeholder) - 1)
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, self._domain_placeholder))

            return loss, weights

    def _build_lstm(self, input_layer, lengths, hidden_units, dropout_prob, n_layers):
        self._lstm_max_len = tf.get_variable('lstm_max_len', dtype=tf.int32, initializer=tf.constant(-1))
        cell = tf.nn.rnn_cell.LSTMCell(hidden_units, forget_bias=1.0, state_is_tuple=True, use_peepholes=True)
        if self._is_train:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1 - dropout_prob)

        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * n_layers, state_is_tuple=True)
        initial_state = cell.zero_state(self._batch_size if self._is_train else 1, tf.float32)
        outputs, state = tf.nn.dynamic_rnn(cell, input_layer, sequence_length=lengths, initial_state=initial_state, parallel_iterations=128)
        # Select just the last output from each example
        outputs = tf.reshape(outputs, (-1, hidden_units))
        indices = (tf.range(self._batch_size) * self._lstm_max_len + tf.to_int32(lengths) - 1)
        outputs = tf.gather(outputs, indices)
        return outputs

    def _load_data(self, data_files, label_map, len_limit=200):
        """Handles data in format created by qanta/guesser/util/format_dan.py"""
        vecs = []
        labels = []
        add_labels = (label_map is None)
        if add_labels:
            label_map = defaultdict()
            label_map.default_factory = label_map.__len__

        complete = []
        weights = []
        domains = []
        max_len = 0
        for file_num, (data_file, domain, weight) in enumerate(data_files):
            if weight == 0:
                continue
            with open(data_file, 'rb') as f:
                log.info('Loading data from {}'.format(data_file))
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
                            if add_labels:
                                label = label_map[l[0]]
                            else:
                                label = label_map.get(l[0], len(label_map))
                            labels.append(label)
                        complete.append(i == len(qs) - 1)
                        weights.append(weight)
                        domains.append(domain)
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
        self._n_classes = len(label_map)
        self._labels = np.array(labels)

        # Conversion of each v to array matters if data is jagged (which it will be for non-training models)
        self._data = np.array([np.array(v) for v in vecs])
        self._lens = np.array(lens)
        self._weights = np.array(weights)
        self._domains = np.array(domains)
        self._complete = complete
        self._max_len = max_len
        self._label_map = dict(label_map)
        if self._is_train:
            indices = len(self._data)
            random.shuffle(indices)
            split_point = int(0.95 * len(indices))
            self._train_indices, self._val_indices = indices[:split_point], indices[split_point]

    def _batches(self, train=True):
        order = [i for i in (self._train_indices if train else self._val_indices)]
        np.random.shuffle(order)
        for indices in (order[i:(i + self._batch_size)] for i in range(0, len(self._data), self._batch_size)):
            if len(indices) == self._batch_size:
                yield self._data[indices, :], self._lens[indices], self._labels[indices], self._weights[indices], self._domains[indices]

    def _run_epoch(self, session, epoch_num, train=True):
        total_loss = 0
        # Reset accuracy accumulators
        session.run(tf.initialize_local_variables())
        start_time = time.time()
        for i, (inputs, lens, labels, weights, domains) in enumerate(self._batches()):
            batch_start = time.time()
            fetches = ((self._loss, self._accuracy_update, self._train_op)
                       if train else
                       (self._loss, self._accuracy_update))
            feed_dict = {self._input_placeholder: inputs,
                         self._len_placeholder: lens,
                         self._label_placeholder: labels}
            if self._use_weights:
                feed_dict[self._weight_placeholder] = weights
            if self._adversarial:
                feed_dict[self._domain_gate_placeholder] = int(not i % self._adversarial_interval)
                feed_dict[self._domain_placeholder] = domains
                fetches += (self._domain_accuracy_update,)

            # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            # run_metadata = tf.RunMetadata()
            # loss, *_ = session.run(fetches, feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
            loss, *_ = session.run(fetches, feed_dict=feed_dict)

            # summary_writer.add_run_metadata(run_metadata, 'step{}'.format(i))
            total_loss += loss
            batch_duration = time.time() - batch_start
            log.info('{} Epoch: {} Batch: {} Loss: {} Duration: {}'.format('Train' if train else 'Val', epoch_num, i, loss, batch_duration))

        accuracy = session.run(self._accuracy)
        avg_loss = total_loss / (i + 1)
        duration = time.time() - start_time
        if self._adversarial:
            domain_accuracy = session.run(self._domain_accuracy)

        return (accuracy, avg_loss, duration) + ((domain_accuracy,) if self._adversarial else ())

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

    def train(self, session, n_epochs):
        if not self._is_train:
            raise ValueError('To use a non-train model, call label() instead')
        session.run(tf.initialize_all_variables())
        if self._lstm_representation:
            session.run(self._lstm_max_len.assign(self._max_len))
        max_accuracy = -1
        for i in range(n_epochs):
            accuracy, avg_loss, duration, *others = self._run_epoch(session, i)
            log.info('Train Epoch: {} Avg loss: {} Accuracy: {}. Ran in {} seconds.'.format(i, avg_loss, accuracy, duration))
            if self._adversarial:
                domain_accuracy = others[0]
                log.info('Domain Accuracy: {}'.format(domain_accuracy))

            val_accuracy, val_loss, val_duration, *others = self._run_epoch(session, i, train=False)
            log.info('Val Epoch: {} Avg loss: {} Accuracy: {}. Ran in {} seconds.'.format(i, val_loss, val_accuracy, val_duration))
            if val_accuracy > max_accuracy:
                max_accuracy = val_accuracy
                log.info('New best accuracy. Saving model')
                self._saver.save(session, DEEP_DAN_PARAMS_TARGET)
                with open(DEEP_TF_PARAMS_TARGET, 'wb') as f:
                    save_vals = {'n_classes': self._n_classes,
                                 'embedding_shape': self._embedding.get_shape(),
                                 'label_map': self._label_map}
                    pickle.dump(save_vals, f)

    def evaluate(self, session):
        """Generate softmax output for all examples in dataset"""
        self._saver.restore(session, DEEP_DAN_PARAMS_TARGET)
        session.run(tf.initialize_local_variables())
        results = []
        count = 0
        for i, (in_array, length, label, complete, domain) in enumerate(zip(self._data, self._lens, self._labels, self._complete, self._domains)):
            if not complete:
                continue
            fetches = (self._softmax_output, self._accuracy_update)
            feed_dict = {self._input_placeholder: in_array, self._len_placeholder: length, self._label_placeholder: label}
            feed_dict = {k: np.expand_dims(v, 0) for k, v in feed_dict.items()}
            if self._lstm_representation:
                session.run(self._lstm_max_len.assign(length))
            softmax_output, _ = session.run(fetches, feed_dict=feed_dict)
            results.append((i, np.squeeze(softmax_output)))
            count += 1
            if count % 1000 == 0:
                log.info('Labeled {} examples'.format(count))
        accuracy = session.run(self._accuracy)
        recalls = self._recall_at_n(results)
        return accuracy, recalls

    def get_representations(self, session):
        self._saver.restore(session, DEEP_DAN_PARAMS_TARGET)
        representations = [[], []]
        count = 0
        for i, (in_array, length, label, complete, domain) in enumerate(
                zip(self._data, self._lens, self._labels, self._complete, self._domains)):
            if i % 10000 == 0:
                log.info('Selected {}/{} representations'.format(count, i))
            # Skip most examples to save time
            if random.random() > (0.02 if domain else 0.001):
                continue
            fetches = (self._representation_layer,)
            feed_dict = {self._input_placeholder: in_array,
                         self._len_placeholder: length}
            feed_dict = {k: np.expand_dims(v, 0) for k, v in feed_dict.items()}
            representations[domain].append(session.run(fetches, feed_dict=feed_dict))
            count += 1

        log.info('Got {} representations'.format(count))
        with open(REPRESENTATION_RES_TARGET, 'wb') as f:
            pickle.dump(representations, f)


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
            train_model = TFDan(
                    data_files=((DEEP_TRAIN_TARGET, True, 1), (DEEP_WIKI_TARGET, False, 1)),
                    # data_files=((DEEP_TRAIN_TARGET, True, 1),),
                    is_train=True,
                    initial_embed=embed)
        log.info('Training model')

        train_model.train(session, n_epochs)


def _load_params():
    with open(DEEP_TF_PARAMS_TARGET, 'rb') as f:
        return pickle.load(f)


def evaluate():
    params = _load_params()
    n_classes = params['n_classes']
    embedding_shape = params['embedding_shape']
    label_map = params['label_map']
    with tf.Graph().as_default(), tf.Session() as session:
        with tf.variable_scope('dan', reuse=None):
            dev_model = TFDan(
                data_files=((DEEP_DEV_TARGET, True, 1),),
                is_train=False,
                initial_embed=None,
                n_classes=n_classes,
                embedding_shape=embedding_shape,
                label_map=label_map)
        log.info('Evaluating model on dev')
        dev_accuracy, dev_recalls = dev_model.evaluate(session)
        log.info('Accuracy on dev: {}'.format(dev_accuracy))
        with open(EVAL_RES_TARGET, 'wb') as f:
            pickle.dump(dev_recalls, f)


def get_representations():
    params = _load_params()
    n_classes = params['n_classes']
    embedding_shape = params['embedding_shape']
    label_map = params['label_map']
    with tf.Graph().as_default(), tf.Session() as session:
        with tf.variable_scope('dan', reuse=None):
            model = TFDan(
                data_files=((DEEP_TRAIN_TARGET, True, 1), (DEEP_WIKI_TARGET, False, 1)),
                is_train=False,
                initial_embed=None,
                n_classes=n_classes,
                embedding_shape=embedding_shape,
                label_map=label_map)
            model.get_representations(session)

if __name__ == '__main__':
    train_dan(50)
