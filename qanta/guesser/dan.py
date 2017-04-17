import pickle
import os
import shutil
from typing import List, Tuple, Optional

from qanta.datasets.abstract import TrainingData, Answer, QuestionText
from qanta.datasets.quiz_bowl import QuizBowlDataset
from qanta.guesser.abstract import AbstractGuesser
from qanta.guesser import nn
from qanta.preprocess import preprocess_dataset, tokenize_question
from qanta.util.io import safe_open, safe_path
from qanta.config import conf
from qanta import logging

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Embedding, BatchNormalization, Activation, Lambda
from keras.losses import sparse_categorical_crossentropy
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, EarlyStopping

import numpy as np


log = logging.get(__name__)

DAN_WE_TMP = '/tmp/qanta/deep/dan_we.pickle'
DAN_WE = 'dan_we.pickle'
DAN_MODEL_TMP_TARGET = '/tmp/qanta/deep/final_dan.keras'
DAN_MODEL_TARGET = 'final_dan.keras'
DAN_PARAMS_TARGET = 'dan_params.pickle'


load_embeddings = nn.create_load_embeddings_function(DAN_WE_TMP, DAN_WE, log)


class DANGuesser(AbstractGuesser):
    def __init__(self):
        super().__init__()
        guesser_conf = conf['guessers']['KerasDAN']
        self.min_answers = guesser_conf['min_answers']
        self.expand_we = guesser_conf['expand_we']
        self.n_hidden_layers = guesser_conf['n_hidden_layers']
        self.n_hidden_units = guesser_conf['n_hidden_units']
        self.dropout_probability = guesser_conf['dropout_probability']
        self.embeddings = None
        self.embedding_lookup = None
        self.max_len = None
        self.i_to_class = None
        self.class_to_i = None
        self.vocab = None
        self.n_classes = None
        self.model = None
        self.max_n_epochs = 100
        self.batch_size = 128
        self.max_patience = 5

    def dump_parameters(self):
        return {
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
            'n_hidden_layers': self.n_hidden_layers,
            'n_hidden_units': self.n_hidden_units,
            'dropout_probability': self.dropout_probability
        }

    def load_parameters(self, params):
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
        self.n_hidden_layers = params['n_hidden_layers']
        self.n_hidden_units = params['n_hidden_units']
        self.dropout_probability = params['dropout_probability']

    def qb_dataset(self):
        return QuizBowlDataset(self.min_answers)

    @classmethod
    def targets(cls) -> List[str]:
        return [DAN_PARAMS_TARGET]

    def build_model(self):
        model = Sequential()
        model.add(Embedding(
            self.embeddings.shape[0],
            output_dim=300,
            mask_zero=True,
            input_length=self.max_len,
            weights=[self.embeddings]
        ))
        model.add(nn.GlobalAveragePooling1DMasked())

        for _ in range(self.n_hidden_layers):
            model.add(Dense(self.n_hidden_units))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dropout(self.dropout_probability))

        model.add(Dense(self.n_classes))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_probability))
        model.add(Activation('softmax'))

        adam = Adam()
        model.compile(
            loss=sparse_categorical_crossentropy, optimizer=adam,
            metrics=['sparse_categorical_accuracy']
        )
        return model

    def train(self, training_data: TrainingData) -> None:
        log.info('Preprocessing training data...')
        x_train, y_train, _, x_test, y_test, _, vocab, class_to_i, i_to_class = preprocess_dataset(training_data)
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

        log.info('Building keras model...')
        self.model = self.build_model()

        log.info('Training model...')
        callbacks = [
            TensorBoard(),
            EarlyStopping(patience=self.max_patience, monitor='val_sparse_categorical_accuracy')
        ]
        history = self.model.fit(
            x_train, y_train,
            validation_data=(x_test, y_test),
            batch_size=self.batch_size, epochs=self.max_n_epochs,
            callbacks=callbacks, verbose=2
        )
        log.info('Done training')
        log.info('Saving model...')
        self.model.save(safe_path(DAN_MODEL_TMP_TARGET))
        log.info('Printing model training history...')
        log.info(history.history)

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
        shutil.copyfile(DAN_MODEL_TMP_TARGET, os.path.join(directory, DAN_MODEL_TARGET))
        with safe_open(os.path.join(directory, DAN_PARAMS_TARGET), 'wb') as f:
            pickle.dump(self.dump_parameters(), f)

    @classmethod
    def load(cls, directory: str):
        guesser = DANGuesser()
        guesser.model = load_model(os.path.join(directory, DAN_MODEL_TARGET))
        with open(os.path.join(directory, DAN_PARAMS_TARGET), 'rb') as f:
            params = pickle.load(f)
            guesser.load_parameters(params)

        return guesser
