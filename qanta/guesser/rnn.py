import pickle
import os
import shutil
from typing import List, Tuple, Optional

from qanta.datasets.abstract import TrainingData, Answer, QuestionText
from qanta.datasets.quiz_bowl import QuizBowlDataset
from qanta.guesser.abstract import AbstractGuesser
from qanta.guesser import nn
from qanta.preprocess import preprocess_dataset, tokenize_question
from qanta.util.io import safe_open
from qanta.config import conf
from qanta import logging

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Embedding, LSTM, GRU, SimpleRNN, BatchNormalization, Activation, Bidirectional
from keras.losses import sparse_categorical_crossentropy
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

import numpy as np


log = logging.get(__name__)

RNN_WE_TMP = '/tmp/qanta/deep/rnn_we.pickle'
RNN_WE = 'rnn_we.pickle'
RNN_MODEL_TMP_TARGET = '/tmp/qanta/deep/final_rnn.keras'
RNN_MODEL_TARGET = 'final_rnn.keras'
RNN_PARAMS_TARGET = 'rnn_params.pickle'


load_embeddings = nn.create_load_embeddings_function(RNN_WE_TMP, RNN_WE, log)


class RNNGuesser(AbstractGuesser):
    def __init__(self):
        super().__init__()
        guesser_conf = conf['guessers']['RNN']
        self.rnn_cell = guesser_conf['rnn_cell']
        self.min_answers = guesser_conf['min_answers']
        self.expand_we = guesser_conf['expand_we']
        self.batch_size = guesser_conf['batch_size']
        self.n_rnn_units = guesser_conf['n_rnn_units']
        self.max_n_epochs = guesser_conf['max_n_epochs']
        self.max_patience = guesser_conf['max_patience']
        self.nn_dropout_rate = guesser_conf['nn_dropout_rate']
        self.bidirectional_rnn = guesser_conf['bidirectional_rnn']
        self.train_on_q_runs = guesser_conf['train_on_q_runs']
        self.n_rnn_layers = guesser_conf['n_rnn_layers']
        self.embeddings = None
        self.embedding_lookup = None
        self.max_len = None
        self.i_to_class = None
        self.class_to_i = None
        self.vocab = None
        self.n_classes = None
        self.model = None
        self.history = None

    def dump_parameters(self):
        return {
            'rnn_cell': self.rnn_cell,
            'min_answers': self.min_answers,
            'embeddings': self.embeddings,
            'embedding_lookup': self.embedding_lookup,
            'max_len': self.max_len,
            'i_to_class': self.i_to_class,
            'class_to_i': self.class_to_i,
            'vocab': self.vocab,
            'n_classes': self.n_classes,
            'max_n_epochs': self.max_n_epochs,
            'batch_size': self.batch_size,
            'max_patience': self.max_patience,
            'n_rnn_units': self.n_rnn_units,
            'nn_dropout_rate': self.nn_dropout_rate,
            'bidirectional_rnn': self.bidirectional_rnn,
            'train_on_q_runs': self.train_on_q_runs,
            'n_rnn_layers': self.n_rnn_layers
        }

    def load_parameters(self, params):
        self.rnn_cell = params['rnn_cell']
        self.min_answers = params['min_answers']
        self.embeddings = params['embeddings']
        self.embedding_lookup = params['embedding_lookup']
        self.max_len = params['max_len']
        self.i_to_class = params['i_to_class']
        self.class_to_i = params['class_to_i']
        self.vocab = params['vocab']
        self.n_classes = params['n_classes']
        self.max_n_epochs = params['max_n_epochs']
        self.batch_size = params['batch_size']
        self.max_patience = params['max_patience']
        self.n_rnn_units = params['n_rnn_units']
        self.nn_dropout_rate = params['nn_dropout_rate']
        self.bidirectional_rnn = params['bidirectional_rnn']
        self.train_on_q_runs = params['train_on_q_runs']
        self.n_rnn_layers = params['n_rnn_layers']

    def parameters(self):
        return {
            'rnn_cell': self.rnn_cell,
            'min_answers': self.min_answers,
            'max_len': self.max_len,
            'n_classes': self.n_classes,
            'max_n_epochs': self.max_n_epochs,
            'batch_size': self.batch_size,
            'max_patience': self.max_patience,
            'n_rnn_units': self.n_rnn_units,
            'nn_dropout_rate': self.nn_dropout_rate,
            'bidirectional_rnn': self.bidirectional_rnn,
            'epcohs_trained_for': np.argmax(self.history['val_sparse_categorical_accuracy']) + 1,
            'best_validation_accuracy': max(self.history['val_sparse_categorical_accuracy']),
            'train_on_q_runs': self.train_on_q_runs,
            'n_rnn_layers': self.n_rnn_layers
        }

    def qb_dataset(self):
        return QuizBowlDataset(self.min_answers)

    @classmethod
    def targets(cls) -> List[str]:
        return [RNN_PARAMS_TARGET]

    def build_model(self):
        if self.rnn_cell == 'lstm':
            cell = LSTM
        elif self.rnn_cell == 'gru':
            cell = GRU
        elif self.rnn_cell == 'simple_rnn':
            cell = SimpleRNN
        else:
            raise ValueError('rnn_cell must be lstm, gru, or simple_rdd and was: {}'.format(self.rnn_cell))
        model = Sequential()
        model.add(Embedding(
            self.embeddings.shape[0],
            self.embeddings.shape[1],
            mask_zero=True,
            input_length=self.max_len,
            weights=[self.embeddings]
        ))
        for _ in range(self.n_rnn_layers):
            if self.bidirectional_rnn:
                model.add(Bidirectional(cell(self.n_rnn_units)))
            else:
                model.add(cell(self.n_rnn_units))
            model.add(Dropout(self.nn_dropout_rate))
        model.add(Dense(self.n_classes))
        model.add(BatchNormalization())
        model.add(Dropout(self.nn_dropout_rate))
        model.add(Activation('softmax'))
        adam = Adam()
        model.compile(
            loss=sparse_categorical_crossentropy, optimizer=adam,
            metrics=['sparse_categorical_accuracy']
        )
        return model

    def train(self, training_data: TrainingData) -> None:
        log.info('Preprocessing training data...')
        x_train, y_train, _, x_test, y_test, _, vocab, class_to_i, i_to_class = preprocess_dataset(
            training_data, create_runs=self.train_on_q_runs)
        self.class_to_i = class_to_i
        self.i_to_class = i_to_class
        self.vocab = vocab

        log.info('Creating embeddings...')
        embeddings, embedding_lookup = load_embeddings(vocab=vocab, expand_glove=self.expand_we, mask_zero=True)
        self.embeddings = embeddings
        self.embedding_lookup = embedding_lookup

        log.info('Converting dataset to embeddings...')
        x_train = [nn.convert_text_to_embeddings_indices(q, embedding_lookup) for q in x_train]
        x_test = [nn.convert_text_to_embeddings_indices(q, embedding_lookup) for q in x_test]
        self.n_classes = nn.compute_n_classes(training_data[1])
        self.max_len = nn.compute_max_len(training_data)
        x_train = np.array(nn.tf_format(x_train, self.max_len, 0))
        x_test = np.array(nn.tf_format(x_test, self.max_len, 0))

        log.info('Building model...')
        self.model = self.build_model()

        log.info('Training model...')
        callbacks = [
            TensorBoard(),
            EarlyStopping(patience=self.max_patience, monitor='val_sparse_categorical_accuracy'),
            ModelCheckpoint(RNN_MODEL_TMP_TARGET, save_best_only=True)
        ]
        history = self.model.fit(
            x_train, y_train,
            validation_data=(x_test, y_test),
            batch_size=self.batch_size, epochs=self.max_n_epochs,
            callbacks=callbacks, verbose=2
        )
        self.history = history.history
        log.info('Done training')

    def guess(self, questions: List[QuestionText], max_n_guesses: Optional[int]) -> List[List[Tuple[Answer, float]]]:
        log.info('Generating {} guesses for each of {} questions'.format(max_n_guesses, len(questions)))
        x_test = [nn.convert_text_to_embeddings_indices(tokenize_question(q), self.embedding_lookup) for q in questions]
        x_test = np.array(nn.tf_format(x_test, self.max_len, 0))
        class_probabilities = self.model.predict_proba(x_test, batch_size=self.batch_size)
        guesses = []
        for row in class_probabilities:
            sorted_labels = np.argsort(-row)[:max_n_guesses]
            sorted_guesses = [self.i_to_class[i] for i in sorted_labels]
            sorted_scores = np.copy(row[sorted_labels])
            guesses.append(list(zip(sorted_guesses, sorted_scores)))
        return guesses

    def save(self, directory: str) -> None:
        shutil.copyfile(RNN_MODEL_TMP_TARGET, os.path.join(directory, RNN_MODEL_TARGET))
        with safe_open(os.path.join(directory, RNN_PARAMS_TARGET), 'wb') as f:
            pickle.dump(self.dump_parameters(), f)

    @classmethod
    def load(cls, directory: str):
        guesser = RNNGuesser()
        guesser.model = load_model(os.path.join(directory, RNN_MODEL_TARGET))
        with open(os.path.join(directory, RNN_PARAMS_TARGET), 'rb') as f:
            params = pickle.load(f)
            guesser.load_parameters(params)

        return guesser
