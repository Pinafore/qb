import pickle
import os
import shutil
from typing import List, Tuple, Optional

from qanta.datasets.abstract import TrainingData, Answer, QuestionText
from qanta.datasets.quiz_bowl import QuizBowlDataset
from qanta.datasets.filtered_wikipedia import FilteredWikipediaDataset
from qanta.guesser.abstract import AbstractGuesser
from qanta.guesser import nn
from qanta.preprocess import preprocess_dataset, tokenize_question
from qanta.util.io import safe_open, safe_path
from qanta.config import conf
from qanta import logging

from keras.models import Model, Sequential, load_model
from keras.layers import (Dense, Dropout, Embedding, BatchNormalization, Activation, Input, Conv1D, MaxPooling1D,
                          Flatten, Concatenate)
from keras.losses import sparse_categorical_crossentropy
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import numpy as np


log = logging.get(__name__)

CNN_WE_TMP = '/tmp/qanta/deep/cnn_we.pickle'
CNN_WE = 'cnn_we.pickle'
CNN_MODEL_TMP_TARGET = '/tmp/qanta/deep/final_cnn.keras'
CNN_MODEL_TARGET = 'final_cnn.keras'
CNN_PARAMS_TARGET = 'cnn_params.pickle'


load_embeddings = nn.create_load_embeddings_function(CNN_WE_TMP, CNN_WE, log)


class CNNGuesser(AbstractGuesser):
    def __init__(self):
        super().__init__()
        guesser_conf = conf['guessers']['CNN']
        self.expand_we = guesser_conf['expand_we']
        self.n_filter_list = guesser_conf['n_filter_list']
        self.filter_sizes = guesser_conf['filter_sizes']
        self.nn_dropout_rate = guesser_conf['nn_dropout_rate']
        self.batch_size = guesser_conf['batch_size']
        self.learning_rate = guesser_conf['learning_rate']
        self.max_n_epochs = guesser_conf['max_n_epochs']
        self.max_patience = guesser_conf['max_patience']
        self.activation_function = guesser_conf['activation_function']
        self.train_on_q_runs = guesser_conf['train_on_q_runs']
        self.train_on_full_q = guesser_conf['train_on_full_q']
        self.decay_lr_on_plateau = guesser_conf['decay_lr_on_plateau']
        self.wiki_data_frac = conf['wiki_data_frac']
        self.max_len = guesser_conf['max_len']
        self.embeddings = None
        self.embedding_lookup = None
        self.i_to_class = None
        self.class_to_i = None
        self.vocab = None
        self.n_classes = None
        self.model = None
        self.history = None

    def dump_parameters(self):
        return {
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
            'n_filter_list': self.n_filter_list,
            'filter_sizes': self.filter_sizes,
            'nn_dropout_rate': self.nn_dropout_rate,
            'learning_rate': self.learning_rate,
            'activation_function': self.activation_function,
            'train_on_q_runs': self.train_on_q_runs,
            'train_on_full_q': self.train_on_full_q,
            'decay_lr_on_plateau': self.decay_lr_on_plateau,
            'wiki_data_frac': self.wiki_data_frac
        }

    def load_parameters(self, params):
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
        self.n_filter_list = params['n_filter_list']
        self.filter_sizes = params['filter_sizes']
        self.nn_dropout_rate = params['nn_dropout_rate']
        self.learning_rate = params['learning_rate']
        self.activation_function = params['activation_function']
        self.train_on_q_runs = params['train_on_q_runs']
        self.train_on_full_q = params['train_on_full_q']
        self.decay_lr_on_plateau = params['decay_lr_on_plateau']
        self.wiki_data_frac = params['wiki_data_frac']

    def parameters(self):
        return {
            'max_len': self.max_len,
            'n_classes': self.n_classes,
            'max_n_epochs': self.max_n_epochs,
            'batch_size': self.batch_size,
            'max_patience': self.max_patience,
            'n_filter_list': self.n_filter_list,
            'filter_sizes': self.filter_sizes,
            'nn_dropout_rate': self.nn_dropout_rate,
            'learning_rate': self.learning_rate,
            'activation_function': self.activation_function,
            'epochs_trained_for': np.argmax(self.history['val_sparse_categorical_accuracy']) + 1,
            'best_validation_accuracy': max(self.history['val_sparse_categorical_accuracy']),
            'train_on_q_runs': self.train_on_q_runs,
            'train_on_full_q': self.train_on_full_q,
            'decay_lr_on_plateau': self.decay_lr_on_plateau,
            'wiki_data_frac': self.wiki_data_frac
        }

    def qb_dataset(self):
        return QuizBowlDataset(1, guesser_train=True)

    @classmethod
    def targets(cls) -> List[str]:
        return [CNN_PARAMS_TARGET]

    def build_model(self):
        model_input = Input((self.max_len,))

        emb_model = Sequential()
        emb_model.add(Embedding(
            self.embeddings.shape[0],
            self.embeddings.shape[1],
            input_length=self.max_len,
            weights=[self.embeddings]
        ))
        emb_model.add(Dropout(self.nn_dropout_rate))

        model = emb_model(model_input)

        for n_filters in self.n_filter_list:
            convolutions = []
            for f_size in self.filter_sizes:
                conv = Conv1D(n_filters, f_size)(model)
                conv = BatchNormalization()(conv)
                conv = Activation(self.activation_function)(conv)
                conv = MaxPooling1D()(conv)
                conv = Flatten()(conv)
                convolutions.append(conv)
            model = Concatenate()(convolutions) if len(convolutions) > 1 else convolutions[0]
        model = Dense(self.n_classes)(model)
        model = BatchNormalization()(model)
        model = Dropout(self.nn_dropout_rate)(model)
        model = Activation('softmax')(model)

        adam = Adam()
        final_model = Model(inputs=[model_input], outputs=model)
        final_model.compile(
            loss=sparse_categorical_crossentropy, optimizer=adam,
            metrics=['sparse_categorical_accuracy']
        )
        return final_model

    def train(self, training_data: TrainingData) -> None:
        log.info('Preprocessing training data...')
        x_train, y_train, x_test, y_test, vocab, class_to_i, i_to_class = preprocess_dataset(
            training_data, create_runs=self.train_on_q_runs, full_question=self.train_on_full_q)
        if self.wiki_data_frac > 0:
            log.info('Using wikipedia with fraction: {}'.format(self.wiki_data_frac))
            wiki_data = FilteredWikipediaDataset().training_data()
            results = preprocess_dataset(
                wiki_data,
                train_size=1,
                vocab=vocab,
                class_to_i=class_to_i,
                i_to_class=i_to_class)
            x_train.extend(results[0])
            y_train.extend(results[1])

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
            EarlyStopping(patience=self.max_patience, monitor='val_sparse_categorical_accuracy'),
            ModelCheckpoint(
                safe_path(CNN_MODEL_TMP_TARGET),
                save_best_only=True,
                monitor='val_sparse_categorical_accuracy'
            )
        ]
        if self.decay_lr_on_plateau:
            callbacks.append(ReduceLROnPlateau(monitor='val_sparse_categorical_accuracy', factor=.5, patience=5))
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
        x_test = [nn.convert_text_to_embeddings_indices(
            tokenize_question(q), self.embedding_lookup)
            for q in questions]
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
        shutil.copyfile(CNN_MODEL_TMP_TARGET, os.path.join(directory, CNN_MODEL_TARGET))
        with safe_open(os.path.join(directory, CNN_PARAMS_TARGET), 'wb') as f:
            pickle.dump(self.dump_parameters(), f)

    @classmethod
    def load(cls, directory: str):
        guesser = CNNGuesser()
        guesser.model = load_model(os.path.join(directory, CNN_MODEL_TARGET))
        with open(os.path.join(directory, CNN_PARAMS_TARGET), 'rb') as f:
            params = pickle.load(f)
            guesser.load_parameters(params)

        return guesser
