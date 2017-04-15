import pickle
import time
import os
import shutil
from typing import Dict, List, Tuple, Union, Optional

from qanta.datasets.abstract import TrainingData
from qanta.datasets.quiz_bowl import QuizBowlDataset
from qanta.datasets.wikipedia import WikipediaDataset
from qanta.guesser.abstract import AbstractGuesser
from qanta.guesser.nn import (make_layer, convert_text_to_embeddings_indices, compute_n_classes,
                              compute_lengths, compute_max_len, tf_format,
                              create_load_embeddings_function, create_batches)
from qanta.preprocess import preprocess_dataset, tokenize_question
from qanta.util.io import safe_open, safe_path, shell
from qanta.config import conf
from qanta import logging

import tensorflow as tf
import numpy as np

log = logging.get(__name__)
TF_DAN_WE_TMP = '/tmp/qanta/deep/tf_dan_we.pickle'
TF_DAN_WE = 'tf_dan_we.pickle'
DEEP_DAN_MODEL_TMP_PREFIX = '/tmp/qanta/deep/tfdan'
DEEP_DAN_MODEL_TMP_DIR = '/tmp/qanta/deep'
DEEP_DAN_MODEL_TARGET = 'tfdan_dir'
DEEP_DAN_PARAMS_TARGET = 'dan_params.pickle'


load_embeddings = create_load_embeddings_function(TF_DAN_WE_TMP, TF_DAN_WE, log)


class TFDanModel:
    def __init__(self, dan_params: Dict, max_len: int, n_classes: int, embeddings, embedding_lookup):
        self.embeddings = embeddings
        self.embedding_lookup = embedding_lookup
        self.dan_params = dan_params
        self.max_len = max_len
        self.n_classes = n_classes
        self.n_hidden_units = dan_params['n_hidden_units']
        self.n_hidden_layers = conf['guessers']['Dan']['n_hidden_layers']
        self.word_dropout = dan_params['word_dropout']
        self.nn_dropout = dan_params['nn_dropout']
        self.batch_size = dan_params['batch_size']
        self.learning_rate = dan_params['learning_rate']
        self.max_epochs = dan_params['max_epochs']
        self.max_patience = dan_params['max_patience']

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
        self.accuracy = None
        self.training_phase = None

        # Set at runtime
        self.summary = None
        self.session = None
        self.summary_counter = 0

    def build_tf_model(self):
        with tf.variable_scope(
                'dan',
                reuse=None,
                initializer=tf.contrib.layers.xavier_initializer()):

            if conf['use_pretrained_embeddings']:
                self.initial_embed = tf.get_variable(
                    'embeddings',
                    initializer=tf.constant(self.embeddings, dtype=tf.float32)
                )
            else:
                self.initial_embed = tf.get_variable(
                    'embeddings', shape=self.embeddings.shape
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
                layer_out, w = make_layer(
                    i, layer_out,
                    n_in=in_dim, n_out=self.n_hidden_units,
                    op=tf.nn.elu, dropout_prob=self.nn_dropout_var,
                    batch_norm=True, batch_is_training=self.training_phase
                )
                in_dim = None
            logits, w = make_layer(self.n_hidden_layers, layer_out, n_out=self.n_classes, op=None)

            with tf.name_scope('cross_entropy'):
                self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits, labels=tf.to_int64(self.label_placeholder))
                self.loss = tf.reduce_mean(self.loss)
                tf.summary.scalar('cross_entropy', self.loss)

            self.softmax_output = tf.nn.softmax(logits)
            preds = tf.to_int32(tf.argmax(logits, 1))

            with tf.name_scope('accuracy'):
                self.batch_accuracy = tf.contrib.metrics.accuracy(preds, self.label_placeholder)
                tf.summary.scalar('accuracy', self.batch_accuracy)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.name_scope('train'):
                with tf.control_dependencies(update_ops):
                    optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                    self.train_op = optimizer.minimize(self.loss)

            self.summary = tf.summary.merge_all()
            self.saver = tf.train.Saver()

    def train(self, x_train, y_train, x_train_lengths, x_test, y_test, x_test_lengths, save=True):
        with tf.Graph().as_default(), tf.Session() as session:
            self.build_tf_model()
            self.session = session
            self.session.run(tf.global_variables_initializer())
            params_suffix = ','.join('{}={}'.format(k, v) for k, v in self.dan_params.items())
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
            fetches = self.loss, self.batch_accuracy, self.train_op
        else:
            fetches = self.loss, self.batch_accuracy, self.summary

        batch_i = 0
        self.session.run(self.word_dropout_var.assign(self.word_dropout if train else 0))
        self.session.run(self.nn_dropout_var.assign(self.nn_dropout if train else 0))
        for x_batch, y_batch, x_len_batch in create_batches(
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
            for x_batch, y_batch, x_len_batch in create_batches(
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
        self.saver.save(self.session, safe_path(DEEP_DAN_MODEL_TMP_PREFIX))

    def load(self):
        self.saver.restore(self.session, DEEP_DAN_MODEL_TMP_PREFIX)


DEFAULT_DAN_PARAMS = dict(
    n_hidden_units=300, word_dropout=.6, batch_size=256,
    learning_rate=.003, max_epochs=100, nn_dropout=0, max_patience=10
)


class DANGuesser(AbstractGuesser):
    def __init__(self, dan_params=DEFAULT_DAN_PARAMS, use_wiki=False):
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
        self.use_wiki = use_wiki
        self.min_answers = conf['guessers']['Dan']['min_appearances']
        self.expand_glove = conf['guessers']['Dan']['expand_glove']

    @classmethod
    def targets(cls) -> List[str]:
        return [DEEP_DAN_PARAMS_TARGET]

    def qb_dataset(self):
        return QuizBowlDataset(self.min_answers)

    def train(self,
              training_data: TrainingData) -> None:
        log.info('Preprocessing training data...')
        x_train, y_train, _, x_test, y_test, _, vocab, class_to_i, i_to_class = preprocess_dataset(training_data)
        self.class_to_i = class_to_i
        self.i_to_class = i_to_class
        self.vocab = vocab

        if self.use_wiki:
            wiki_training_data = WikipediaDataset(self.min_answers).training_data()
            x_train_wiki, y_train_wiki, _, _, _, _, _, _, _ = preprocess_dataset(
                wiki_training_data, train_size=1, vocab=vocab, class_to_i=class_to_i,
                i_to_class=i_to_class)

        log.info('Creating embeddings...')
        embeddings, embedding_lookup = load_embeddings(vocab=vocab, expand_glove=self.expand_glove)
        self.embeddings = embeddings
        self.embedding_lookup = embedding_lookup

        log.info('Converting dataset to embeddings...')
        x_train = [convert_text_to_embeddings_indices(q, embedding_lookup) for q in x_train]
        x_train_lengths = compute_lengths(x_train)

        x_test = [convert_text_to_embeddings_indices(q, embedding_lookup) for q in x_test]
        x_test_lengths = compute_lengths(x_test)

        if self.use_wiki:
            x_train_wiki = [convert_text_to_embeddings_indices(q, embedding_lookup) for q in x_train_wiki]
            x_train_lengths_wiki = compute_lengths(x_train_wiki)
            x_train.extend(x_train_wiki)
            y_train.extend(y_train_wiki)
            x_train_lengths = np.concatenate([x_train_lengths, x_train_lengths_wiki])

        log.info('Computing number of classes and max paragraph length in words')
        self.n_classes = compute_n_classes(training_data[1])
        self.max_len = compute_max_len(training_data)
        x_train = tf_format(x_train, self.max_len, embeddings.shape[0])
        x_test = tf_format(x_test, self.max_len, embeddings.shape[0])

        log.info('Training deep model...')
        self.model = TFDanModel(self.dan_params, self.max_len, self.n_classes, self.embeddings, self.embedding_lookup)
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        self.model.train(x_train, y_train, x_train_lengths, x_test, y_test, x_test_lengths)

    def guess(self, questions: List[str], n_guesses: Optional[int]) -> List[List[Tuple[str, float]]]:
        log.info('Generating {} guesses for each of {} questions'.format(n_guesses, len(questions)))
        log.info('Converting text to embedding indices...')
        x_test = [convert_text_to_embeddings_indices(
            tokenize_question(q), self.embedding_lookup) for q in questions]
        log.info('Computing question lengths...')
        x_test_lengths = compute_lengths(x_test)
        log.info('Converting questions to tensorflow format...')
        x_test = tf_format(x_test, self.max_len, self.embeddings.shape[0])
        x_test = np.array(x_test)
        self.model = TFDanModel(self.dan_params, self.max_len, self.n_classes, self.embeddings, self.embedding_lookup)
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
        return {
            **self.dan_params,
            'use_wiki': self.use_wiki,
            'min_answers': self.min_answers,
            'expand_glove': self.expand_glove
        }

    @classmethod
    def load(cls, directory: str) -> AbstractGuesser:
        guesser = DANGuesser()
        embeddings, embedding_lookup = load_embeddings(root_directory=directory)
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
        shell('cp -r {} {}'.format(model_path, safe_path(DEEP_DAN_MODEL_TMP_DIR)))

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
