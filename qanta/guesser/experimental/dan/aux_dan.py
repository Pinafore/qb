import pickle
import time
import os
import shutil
from typing import Dict, List, Tuple, Union, Optional

from qanta.datasets.abstract import TrainingData
from qanta.guesser.abstract import AbstractGuesser
from qanta.guesser.dan.util import (make_layer, load_embeddings, create_batches,
                                    convert_text_to_embeddings_indices, compute_lengths,
                                    compute_n_classes, compute_max_len, tf_format,
                                    compute_ans_type_classes, compute_category_classes,
                                    compute_gender_classes)
from qanta.datasets.quiz_bowl import QuizBowlDataset
from qanta.datasets.wikipedia import WikipediaDataset
from qanta.preprocess import preprocess_dataset, tokenize_question
from qanta.util.io import safe_open, safe_path, shell
from qanta import logging

import tensorflow as tf
import numpy as np

log = logging.get(__name__)
AUX_DAN_WE_TMP = '/tmp/qanta/deep/aux_dan_we.pickle'
AUX_DAN_WE = 'aux_dan_we.pickle'
DEEP_DAN_AUX_MODEL_TMP_PREFIX = '/tmp/qanta/deep/aux_dan'
DEEP_DAN_AUX_MODEL_TMP_DIR = '/tmp/qanta/deep'
DEEP_DAN_AUX_MODEL_TARGET = 'auxdan_dir'
DEEP_DAN_AUX_PARAMS_TARGET = 'aux_dan_params.pickle'


def _load_embeddings(vocab=None, root_directory=''):
    return load_embeddings(AUX_DAN_WE_TMP, AUX_DAN_WE, vocab=vocab, root_directory=root_directory)


class AuxDanModel:
    def __init__(self, dan_params: Dict, max_len: int, n_classes: int, n_ans_type_classes: int,
                 n_gender_classes: int, n_category_classes: int):
        self.dan_params = dan_params
        self.max_len = max_len
        self.n_classes = n_classes
        self.n_ans_type_classes = n_ans_type_classes
        self.n_gender_classes = n_gender_classes
        self.n_category_classes = n_category_classes
        self.n_hidden_units = dan_params['n_hidden_units']
        self.n_hidden_layers = dan_params['n_hidden_layers']
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
        self.ans_type_label_placeholder = None
        self.category_label_placeholder = None
        self.gender_label_placeholder = None

        self.guess_loss = None
        self.ans_type_loss = None
        self.category_loss = None
        self.gender_loss = None
        self.loss = None

        self.guess_batch_accuracy = None
        self.ans_type_batch_accuracy = None
        self.category_batch_accuracy = None
        self.gender_batch_accuracy = None

        self.train_op = None

        self.guess_softmax_output = None
        self.ans_type_softmax_output = None
        self.category_softmax_output = None
        self.gender_softmax_output = None

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

        # Set at runtime
        self.training_phase = None
        self.summary = None
        self.session = None
        self.summary_counter = 0
        self.all_batch_counter = 0

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

            self.ans_type_label_placeholder = tf.placeholder(tf.int32, shape=self.batch_size,
                                                             name='ans_type_label_placeholder')
            self.category_label_placeholder = tf.placeholder(tf.int32, shape=self.batch_size,
                                                             name='category_label_placeholder')
            self.gender_label_placeholder = tf.placeholder(tf.int32, shape=self.batch_size,
                                                           name='gender_label_placeholder')

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
                layer_out, _, _ = make_layer(
                    str(i), layer_out,
                    n_in=in_dim, n_out=self.n_hidden_units,
                    op=tf.nn.elu, dropout_prob=self.nn_dropout_var,
                    batch_norm=True, batch_is_training=self.training_phase
                )
                in_dim = None

            ans_type_logits, _, _ = make_layer('ans_type', layer_out,
                                                     n_out=self.n_ans_type_classes, op=None)
            category_logits, _, _ = make_layer('category', layer_out,
                                                     n_out=self.n_category_classes, op=None)
            gender_logits, _, _ = make_layer('gender', layer_out,
                                                 n_out=self.n_gender_classes, op=None)

            concat_layer = tf.concat(
                [ans_type_logits, category_logits, gender_logits, layer_out], 1)
            combined_layer, _, _ = make_layer('combined', concat_layer,
                                              n_out=concat_layer.get_shape()[1], op=tf.nn.elu)
            guess_logits, _, _ = make_layer('label', combined_layer, n_out=self.n_classes, op=None)

            with tf.name_scope('cross_entropy'):
                self.guess_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=guess_logits, labels=tf.to_int64(self.label_placeholder))
                self.guess_loss = tf.reduce_mean(self.guess_loss) / 2
                tf.summary.scalar('guess_loss', self.guess_loss)

                self.ans_type_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=ans_type_logits, labels=tf.to_int64(self.ans_type_label_placeholder)
                )
                self.ans_type_loss = tf.reduce_mean(self.ans_type_loss) / 6
                tf.summary.scalar('ans_type_loss', self.ans_type_loss)

                self.category_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=category_logits, labels=tf.to_int64(self.category_label_placeholder)
                )
                self.category_loss = tf.reduce_mean(self.category_loss) / 6
                tf.summary.scalar('category_type_loss', self.category_loss)

                self.gender_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=gender_logits, labels=tf.to_int64(self.gender_label_placeholder)
                )
                self.gender_loss = tf.reduce_mean(self.gender_loss) / 6
                tf.summary.scalar('gender_loss', self.gender_loss)

                self.loss = (self.guess_loss + self.ans_type_loss + self.category_loss
                             + self.gender_loss)

            self.guess_softmax_output = tf.nn.softmax(guess_logits)
            guess_preds = tf.to_int32(tf.argmax(guess_logits, 1))

            self.ans_type_softmax_output = tf.nn.softmax(ans_type_logits)
            ans_type_preds = tf.to_int32(tf.argmax(ans_type_logits, 1))

            self.category_softmax_output = tf.nn.softmax(category_logits)
            category_preds = tf.to_int32(tf.argmax(category_logits, 1))

            self.gender_softmax_output = tf.nn.softmax(gender_logits)
            gender_preds = tf.to_int32(tf.argmax(gender_logits, 1))

            with tf.name_scope('accuracy'):
                self.guess_batch_accuracy = tf.contrib.metrics.accuracy(
                    guess_preds, self.label_placeholder)
                tf.summary.scalar('guess_accuracy', self.guess_batch_accuracy)

                self.ans_type_batch_accuracy = tf.contrib.metrics.accuracy(
                    ans_type_preds, self.ans_type_label_placeholder)
                tf.summary.scalar('ans_type_accuracy', self.ans_type_batch_accuracy)

                self.category_batch_accuracy = tf.contrib.metrics.accuracy(
                    category_preds, self.category_label_placeholder)
                tf.summary.scalar('category_accuracy', self.category_batch_accuracy)

                self.gender_batch_accuracy = tf.contrib.metrics.accuracy(
                    gender_preds, self.gender_label_placeholder)
                tf.summary.scalar('gender_accuracy', self.gender_batch_accuracy)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.name_scope('train'):
                with tf.control_dependencies(update_ops):
                    optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                    self.train_op = optimizer.minimize(self.loss)

            self.summary = tf.summary.merge_all()
            self.saver = tf.train.Saver()

    def train(self,
              x_train, y_train, x_train_lengths, property_labels_train,
              x_test, y_test, x_test_lengths, property_labels_test, save=True):
        with tf.Graph().as_default(), tf.Session() as session:
            self.build_tf_model()
            self.session = session
            self.session.run(tf.global_variables_initializer())
            params_suffix = ','.join('{}={}'.format(k, v) for k, v in self.dan_params.items())
            self.file_writer = tf.summary.FileWriter(
                os.path.join('output/tensorflow', params_suffix), session.graph)
            train_losses, train_accuracies, holdout_losses, holdout_accuracies = self._train(
                x_train, y_train, x_train_lengths, property_labels_train,
                x_test, y_test, x_test_lengths, property_labels_test,
                self.max_epochs, save=save
            )

            return train_losses, train_accuracies, holdout_losses, holdout_accuracies

    def _train(self,
               x_train, y_train, x_train_lengths, property_labels_train,
               x_test, y_test, x_test_lengths, property_labels_test,
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
                x_train, y_train, x_train_lengths, property_labels_train
            )
            log.info(
                'Train Epoch: {} Avg loss: {:.4f} Accuracy: {:.4f}. Ran in {:.4f} seconds.'.format(
                    i, np.average(losses), np.average(accuracies), duration))
            train_accuracies.append(accuracies)
            train_losses.append(losses)

            # Validation Epoch
            val_accuracies, val_losses, val_duration = self.run_epoch(
                x_test, y_test, x_test_lengths, property_labels_test, train=False
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

    def run_epoch(self, x_data, y_data, x_lengths, property_labels, train=True):
        start_time = time.time()
        accuracies = []
        losses = []
        if train:
            fetches = {
                self.loss: self.loss,
                self.guess_batch_accuracy: self.guess_batch_accuracy,
                self.ans_type_batch_accuracy: self.ans_type_batch_accuracy,
                self.category_batch_accuracy: self.category_batch_accuracy,
                self.gender_batch_accuracy: self.gender_batch_accuracy,
                self.train_op: self.train_op
            }
        else:
            fetches = {
                self.loss: self.loss,
                self.guess_batch_accuracy: self.guess_batch_accuracy,
                self.ans_type_batch_accuracy: self.ans_type_batch_accuracy,
                self.category_batch_accuracy: self.category_batch_accuracy,
                self.gender_batch_accuracy: self.gender_batch_accuracy,
                self.summary: self.summary
            }

        batch_i = 0
        self.session.run(self.word_dropout_var.assign(self.word_dropout if train else 0))
        self.session.run(self.nn_dropout_var.assign(self.nn_dropout if train else 0))
        ans_type_labels = property_labels['ans_type']
        category_labels = property_labels['category']
        gender_labels = property_labels['gender']
        for x_batch, y_batch, x_len_batch, ans_type_batch, category_batch, gender_batch in create_batches(
                self.batch_size, x_data, y_data, x_lengths,
                ans_type_labels, category_labels, gender_labels):
            feed_dict = {
                self.input_placeholder: x_batch,
                self.label_placeholder: y_batch,
                self.len_placeholder: x_len_batch,
                self.training_phase: int(train),
                self.ans_type_label_placeholder: ans_type_batch,
                self.category_label_placeholder: category_batch,
                self.gender_label_placeholder: gender_batch
            }
            if self.all_batch_counter % 100 == 99:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                returned = self.session.run(fetches, feed_dict=feed_dict, options=run_options,
                                            run_metadata=run_metadata)
                self.file_writer.add_run_metadata(run_metadata, 'step{}'.format(
                    self.all_batch_counter))
            else:
                returned = self.session.run(fetches, feed_dict=feed_dict)
            loss = returned[self.loss]
            accuracy = returned[self.guess_batch_accuracy]
            if not train:
                summary = returned[self.summary]
                self.file_writer.add_summary(summary, self.summary_counter)
                self.summary_counter += 1

            accuracies.append(accuracy)
            losses.append(loss)
            batch_i += 1
            self.all_batch_counter += 1
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
            for x_batch, y_batch, x_len_batch, ans_type_batch, category_batch, gender_batch in create_batches(
                    self.batch_size, x_test, y_test, x_test_lengths,
                    y_test, y_test, y_test,
                    pad=True, shuffle=False):
                if batch_i % 250 == 0:
                    log.info('Starting batch {}'.format(batch_i))
                feed_dict = {
                    self.input_placeholder: x_batch,
                    self.label_placeholder: y_batch,
                    self.len_placeholder: x_len_batch,
                    self.training_phase: 0,
                    self.ans_type_label_placeholder: ans_type_batch,
                    self.category_label_placeholder: category_batch,
                    self.gender_label_placeholder: gender_batch
                }
                batch_predictions = self.session.run(self.guess_softmax_output, feed_dict=feed_dict)
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
        self.saver.save(self.session, safe_path(DEEP_DAN_AUX_MODEL_TMP_PREFIX))

    def load(self):
        self.saver.restore(self.session, DEEP_DAN_AUX_MODEL_TMP_PREFIX)


DEFAULT_AUX_DAN_PARAMS = dict(
    n_hidden_units=200, n_hidden_layers=2, word_dropout=.6, batch_size=128,
    learning_rate=.001, max_epochs=100, nn_dropout=0, max_patience=10
)


class AuxDANGuesser(AbstractGuesser):
    def __init__(self, dan_params=DEFAULT_AUX_DAN_PARAMS, use_wiki=False, min_answers=1):
        super().__init__()
        self.dan_params = dan_params
        self.model = None  # type: Union[None, AuxDanModel]
        self.embedding_lookup = None
        self.max_len = None  # type: Union[None, int]
        self.embeddings = None

        self.i_to_class = None
        self.class_to_i = None

        self.ans_type_i_to_class = None
        self.ans_type_class_to_i = None

        self.category_i_to_class = None
        self.category_class_to_i = None

        self.gender_i_to_class = None
        self.gender_class_to_i = None

        self.vocab = None

        self.n_classes = None
        self.n_ans_type_classes = None
        self.n_category_classes = None
        self.n_gender_classes = None

        self.use_wiki = use_wiki
        self.min_answers = min_answers

    @classmethod
    def targets(cls) -> List[str]:
        return [DEEP_DAN_AUX_PARAMS_TARGET]

    def qb_dataset(self):
        return QuizBowlDataset(self.min_answers)

    def train(self, training_data: TrainingData) -> None:
        log.info('Preprocessing training data...')
        x_train, y_train, properties_train, x_test, y_test, properties_test, vocab,\
            class_to_i, i_to_class = preprocess_dataset(training_data)

        self.class_to_i = class_to_i
        self.i_to_class = i_to_class
        self.vocab = vocab

        self.ans_type_i_to_class, self.ans_type_class_to_i = compute_ans_type_classes(
            properties_train)
        self.n_ans_type_classes = len(self.ans_type_class_to_i)
        compute_ans_type_classes(properties_test)

        self.category_i_to_class, self.category_class_to_i = compute_category_classes(
            properties_train)
        self.n_category_classes = len(self.category_class_to_i)
        compute_category_classes(properties_test)

        self.gender_i_to_class, self.gender_class_to_i = compute_gender_classes(properties_train)
        self.n_gender_classes = len(self.gender_class_to_i)
        compute_gender_classes(properties_test)

        ans_type_labels_train = np.array([self.ans_type_class_to_i[prop['ans_type']]
                                          for prop in properties_train])
        category_labels_train = np.array([self.category_class_to_i[prop['category']]
                                          for prop in properties_train])
        gender_labels_train = np.array([self.gender_class_to_i[prop['gender']]
                                        for prop in properties_train])

        ans_type_labels_test = np.array([self.ans_type_class_to_i[prop['ans_type']]
                                         for prop in properties_test])
        category_labels_test = np.array([self.category_class_to_i[prop['category']]
                                         for prop in properties_test])
        gender_labels_test = np.array([self.gender_class_to_i[prop['gender']]
                                       for prop in properties_test])

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
        x_train = [convert_text_to_embeddings_indices(q, embedding_lookup) for q in x_train]
        x_train_lengths = compute_lengths(x_train)

        x_test = [convert_text_to_embeddings_indices(q, embedding_lookup) for q in x_test]
        x_test_lengths = compute_lengths(x_test)

        log.info('Computing number of classes and max paragraph length in words')
        self.n_classes = compute_n_classes(training_data[1])
        self.max_len = compute_max_len(x_train)
        x_train = tf_format(x_train, self.max_len, embeddings.shape[0])
        x_test = tf_format(x_test, self.max_len, embeddings.shape[0])

        log.info('Training deep model...')
        self.model = AuxDanModel(
            self.dan_params, self.max_len, self.n_classes,
            self.n_ans_type_classes, self.n_gender_classes, self.n_category_classes
        )
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        property_labels_train = {
            'ans_type': ans_type_labels_train,
            'category': category_labels_train,
            'gender': gender_labels_train
        }

        property_labels_test = {
            'ans_type': ans_type_labels_test,
            'category': category_labels_test,
            'gender': gender_labels_test
        }
        train_losses, train_accuracies, holdout_losses, holdout_accuracies = self.model.train(
            x_train, y_train, x_train_lengths, property_labels_train,
            x_test, y_test, x_test_lengths, property_labels_test)

    def guess(self,
              questions: List[str], n_guesses: Optional[int]) -> List[List[Tuple[str, float]]]:
        log.info('Generating {} guesses for each of {} questions'.format(n_guesses, len(questions)))
        log.info('Converting text to embedding indices...')
        x_test = [convert_text_to_embeddings_indices(
            tokenize_question(q), self.embedding_lookup) for q in questions]
        log.info('Computing question lengths...')
        x_test_lengths = compute_lengths(x_test)
        log.info('Converting questions to tensorflow format...')
        x_test = tf_format(x_test, self.max_len, self.embeddings.shape[0])
        x_test = np.array(x_test)
        self.model = AuxDanModel(
            self.dan_params, self.max_len, self.n_classes, self.n_ans_type_classes,
            self.n_gender_classes, self.n_category_classes
        )
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

    def display_name(self) -> str:
        return 'AuxDAN'

    def parameters(self):
        return {**self.dan_params, 'use_wiki': self.use_wiki, 'min_answers': self.min_answers}

    @classmethod
    def load(cls, directory: str) -> AbstractGuesser:
        guesser = AuxDANGuesser()
        embeddings, embedding_lookup = _load_embeddings(root_directory=directory)
        guesser.embeddings = embeddings
        guesser.embedding_lookup = embedding_lookup
        params_path = os.path.join(directory, DEEP_DAN_AUX_PARAMS_TARGET)
        with open(params_path, 'rb') as f:
            params = pickle.load(f)
            guesser.max_len = params['max_len']
            guesser.class_to_i = params['class_to_i']
            guesser.i_to_class = params['i_to_class']
            guesser.vocab = params['vocab']
            guesser.n_classes = params['n_classes']
            guesser.n_ans_type_classes = params['n_ans_type_classes']
            guesser.n_category_classes = params['n_category_classes']
            guesser.n_gender_classes = params['n_gender_classes']

            guesser.ans_type_class_to_i = params['ans_type_class_to_i']
            guesser.category_class_to_i = params['category_class_to_i']
            guesser.gender_class_to_i = params['gender_class_to_i']

            guesser.ans_type_i_to_class = params['ans_type_i_to_class']
            guesser.category_i_to_class = params['category_i_to_class']
            guesser.gender_i_to_class = params['gender_i_to_class']

            if (guesser.max_len is None
                    or guesser.class_to_i is None
                    or guesser.i_to_class is None
                    or guesser.ans_type_class_to_i is None
                    or guesser.category_class_to_i is None
                    or guesser.gender_class_to_i is None
                    or guesser.ans_type_i_to_class is None
                    or guesser.category_i_to_class is None
                    or guesser.gender_i_to_class is None
                    or guesser.vocab is None
                    or guesser.n_ans_type_classes is None
                    or guesser.n_category_classes is None
                    or guesser.n_gender_classes is None
                    or guesser.n_classes is None):
                raise ValueError('Attempting to load uninitialized model parameters')
        model_path = os.path.join(directory, DEEP_DAN_AUX_MODEL_TARGET)
        shell('cp -r {} {}'.format(model_path, safe_path(DEEP_DAN_AUX_MODEL_TMP_DIR)))

        we_path = os.path.join(directory, AUX_DAN_WE)
        shutil.copyfile(AUX_DAN_WE_TMP, we_path)

        return guesser

    def save(self, directory: str) -> None:
        params_path = os.path.join(directory, DEEP_DAN_AUX_PARAMS_TARGET)
        with safe_open(params_path, 'wb') as f:
            if (self.max_len is None
                    or self.class_to_i is None
                    or self.i_to_class is None
                    or self.ans_type_class_to_i is None
                    or self.category_class_to_i is None
                    or self.gender_class_to_i is None
                    or self.ans_type_i_to_class is None
                    or self.category_i_to_class is None
                    or self.gender_i_to_class is None
                    or self.vocab is None
                    or self.n_ans_type_classes is None
                    or self.n_category_classes is None
                    or self.n_gender_classes is None
                    or self.n_classes is None):
                raise ValueError('Attempting to save uninitialized model parameters')
            pickle.dump({
                'max_len': self.max_len,
                'class_to_i': self.class_to_i,
                'i_to_class': self.i_to_class,
                'vocab': self.vocab,
                'n_classes': self.n_classes,
                'n_ans_type_classes': self.n_ans_type_classes,
                'n_category_classes': self.n_category_classes,
                'n_gender_classes': self.n_gender_classes,
                'ans_type_class_to_i': self.ans_type_class_to_i,
                'category_class_to_i': self.category_class_to_i,
                'gender_class_to_i': self.gender_class_to_i,
                'ans_type_i_to_class': self.ans_type_i_to_class,
                'category_i_to_class': self.category_i_to_class,
                'gender_i_to_class': self.gender_i_to_class
            }, f)
        model_path = os.path.join(directory, DEEP_DAN_AUX_MODEL_TARGET)
        shell('cp -r {} {}'.format(DEEP_DAN_AUX_MODEL_TMP_DIR, safe_path(model_path)))
        we_path = os.path.join(directory, AUX_DAN_WE)
        shutil.copyfile(AUX_DAN_WE_TMP, safe_path(we_path))
