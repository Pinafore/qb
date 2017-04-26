import pickle
import os
import shutil
from typing import List, Tuple, Optional, Dict
import nltk

from qanta.datasets.abstract import TrainingData, Answer, QuestionText
from qanta.datasets.quiz_bowl import QuizBowlDataset
from qanta.guesser.abstract import AbstractGuesser
from qanta.guesser import nn
from qanta.preprocess import preprocess_dataset, tokenize_question, format_guess
from qanta.util.io import safe_open, safe_path
from qanta.config import conf
from qanta.keras import AverageWords, WordDropout
from qanta.spark import create_spark_context
from qanta.wikipedia.cached_wikipedia import CachedWikipedia
from qanta import logging

from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Embedding, BatchNormalization, Activation, Reshape, Input, dot, multiply, add
from keras.losses import sparse_categorical_crossentropy
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences

import elasticsearch
from elasticsearch_dsl import DocType, Text, Search, Index
from elasticsearch_dsl.connections import connections
import progressbar

import numpy as np


log = logging.get(__name__)
connections.create_connection(hosts=['localhost'])

MEM_QB_WE_TMP = '/tmp/qanta/deep/mem_nn_qb_we.pickle'
MEM_QB_WE = 'mem_nn_qb_we.pickle'
MEM_WIKI_WE_TMP = '/tmp/qanta/deep/mem_nn_wiki_we.pickle'
MEM_WIKI_WE = 'mem_nn_wiki_we.pickle'
MEM_MODEL_TMP_TARGET = '/tmp/qanta/deep/mem_nn.h5'
MEM_MODEL_TARGET = 'mem_nn.h5'
MEM_PARAMS_TARGET = 'mem_nn_params.pickle'


qb_load_embeddings = nn.create_load_embeddings_function(MEM_QB_WE_TMP, MEM_QB_WE, log)
wiki_load_embeddings = nn.create_load_embeddings_function(MEM_WIKI_WE_TMP, MEM_WIKI_WE, log)


class Memory(DocType):
    # Currently this is just for debugging purposes, it isn't used
    page = Text()
    text = Text()

    class Meta:
        index = 'mem_nn'


class MemoryIndex:
    @staticmethod
    def build(documents: Dict[str, List[str]]):
        try:
            Index('mem_nn').delete()
        except elasticsearch.exceptions.NotFoundError:
            log.info('Could not delete non-existent index, continuing to creating new index...')
        Memory.init()
        bar = progressbar.ProgressBar()
        for page in bar(documents):
            for m_text in documents[page]:
                memory = Memory(page=page.replace('_', ' '), text=m_text)
                memory.save()

    @staticmethod
    def search(text: str, max_n_memories: int, include_page=False):
        s = Search(index='mem_nn')[0:max_n_memories].query(
            'multi_match', query=text, fields=['text', 'page']
        )
        if include_page:
            return [(r.text, r.page) for r in s.execute()]
        else:
            return [r.text for r in s.execute()]

    @staticmethod
    def search_parallel(text_list: List[List[str]], max_n_memories: int, include_page=False):
        n_cores = conf['guessers']['MemNN']['n_cores']
        sc = create_spark_context(configs=[('spark.executor.cores', n_cores), ('spark.executor.memory', '4g')])

        def es_search(query: List[str]):
            return MemoryIndex.search(' '.join(query), max_n_memories, include_page=include_page)

        return sc.parallelize(text_list, 400 * n_cores).map(es_search).collect()


def build_wikipedia_sentences(pages, n_sentences):
    cw = CachedWikipedia()
    page_sentences = {}
    for p in pages:
        page_sentences[p] = nltk.tokenize.sent_tokenize(cw[p].content)[:n_sentences]
    return page_sentences


def build_qb_sentences(training_data: TrainingData):
    page_sentences = {}
    for sentences, page in zip(training_data[0], training_data[1]):
        if page in page_sentences:
            page_sentences[page].extend(sentences)
        else:
            page_sentences[page] = sentences

    return page_sentences

Sentence = str
NMemories = List[Sentence]


def preprocess_wikipedia(question_memories: List[NMemories], create_vocab):
    if create_vocab:
        vocab = set()
    else:
        vocab = None

    tokenized_question_memories = []
    for memories in question_memories:
        tokenized_memories = []
        for mem in memories:
            words = tokenize_question(mem)
            tokenized_memories.append(words)
            if create_vocab:
                for w in words:
                    vocab.add(w)
        tokenized_question_memories.append(tokenized_memories)

    return tokenized_question_memories, vocab


class MemNNGuesser(AbstractGuesser):
    def __init__(self):
        super().__init__()
        guesser_conf = conf['guessers']['MemNN']
        self.min_answers = guesser_conf['min_answers']
        self.expand_we = guesser_conf['expand_we']
        self.n_hops = guesser_conf['n_hops']
        self.n_hidden_units = guesser_conf['n_hidden_units']
        self.nn_dropout_rate = guesser_conf['nn_dropout_rate']
        self.word_dropout_rate = guesser_conf['word_dropout_rate']
        self.batch_size = guesser_conf['batch_size']
        self.learning_rate = guesser_conf['learning_rate']
        self.l2_normalize_averaged_words = guesser_conf['l2_normalize_averaged_words']
        self.max_n_epochs = guesser_conf['max_n_epochs']
        self.max_patience = guesser_conf['max_patience']
        self.activation_function = guesser_conf['activation_function']
        self.n_memories = guesser_conf['n_memories']
        self.n_wiki_sentences = guesser_conf['n_wiki_sentences']
        self.qb_embeddings = None
        self.qb_embedding_lookup = None
        self.wiki_embeddings = None
        self.wiki_embedding_lookup = None
        self.qb_max_len = None
        self.wiki_max_len = None
        self.i_to_class = None
        self.class_to_i = None
        self.qb_vocab = None
        self.wiki_vocab = None
        self.n_classes = None
        self.model = None

    def dump_parameters(self):
        return {
            'min_answers': self.min_answers,
            'qb_embeddings': self.qb_embeddings,
            'qb_embedding_lookup': self.qb_embedding_lookup,
            'wiki_embeddings': self.wiki_embeddings,
            'wiki_embedding_lookup': self.wiki_embedding_lookup,
            'qb_max_len': self.qb_max_len,
            'wiki_max_len': self.wiki_max_len,
            'i_to_class': self.i_to_class,
            'class_to_i': self.class_to_i,
            'qb_vocab': self.qb_vocab,
            'wiki_vocab': self.wiki_vocab,
            'n_classes': self.n_classes,
            'max_n_epochs': self.max_n_epochs,
            'batch_size': self.batch_size,
            'max_patience': self.max_patience,
            'n_hops': self.n_hops,
            'n_hidden_units': self.n_hidden_units,
            'nn_dropout_rate': self.nn_dropout_rate,
            'word_dropout_rate': self.word_dropout_rate,
            'learning_rate': self.learning_rate,
            'l2_normalize_averaged_words': self.l2_normalize_averaged_words,
            'activation_function': self.activation_function,
            'n_memories': self.n_memories,
            'n_wiki_sentences': self.n_wiki_sentences
        }

    def load_parameters(self, params):
        self.min_answers = params['min_answers']
        self.qb_embeddings = params['qb_embeddings']
        self.qb_embedding_lookup = params['qb_embedding_lookup']
        self.wiki_embeddings = params['wiki_embeddings']
        self.wiki_embedding_lookup = params['wiki_embedding_lookup']
        self.qb_max_len = params['qb_max_len']
        self.wiki_max_len = params['wiki_max_len']
        self.i_to_class = params['i_to_class']
        self.class_to_i = params['class_to_i']
        self.qb_vocab = params['qb_vocab']
        self.wiki_vocab = params['wiki_vocab']
        self.n_classes = params['n_classes']
        self.max_n_epochs = params['max_n_epochs']
        self.batch_size = params['batch_size']
        self.max_patience = params['max_patience']
        self.n_hops = params['n_hops']
        self.n_hidden_units = params['n_hidden_units']
        self.nn_dropout_rate = params['nn_dropout_rate']
        self.word_dropout_rate = params['word_dropout_rate']
        self.l2_normalize_averaged_words = params['l2_normalize_averaged_words']
        self.learning_rate = params['learning_rate']
        self.activation_function = params['activation_function']
        self.n_memories = params['n_memories']
        self.n_wiki_sentences = params['n_wiki_sentences']

    def parameters(self):
        return {
            'min_answers': self.min_answers,
            'qb_max_len': self.qb_max_len,
            'wiki_max_len': self.wiki_max_len,
            'n_classes': self.n_classes,
            'max_n_epochs': self.max_n_epochs,
            'batch_size': self.batch_size,
            'max_patience': self.max_patience,
            'n_hops': self.n_hops,
            'n_hidden_units': self.n_hidden_units,
            'nn_dropout_rate': self.nn_dropout_rate,
            'word_dropout_rate': self.word_dropout_rate,
            'learning_rate': self.learning_rate,
            'l2_normalize_averaged_words': self.l2_normalize_averaged_words,
            'activation_function': self.activation_function,
            'n_memories': self.n_memories,
            'n_wiki_sentences': self.n_wiki_sentences
        }

    def qb_dataset(self):
        return QuizBowlDataset(self.min_answers)

    @classmethod
    def targets(cls) -> List[str]:
        return [MEM_PARAMS_TARGET]

    def build_model(self):
        wiki_vocab_size = self.wiki_embeddings.shape[0]
        wiki_we_dimension = self.wiki_embeddings.shape[1]
        qb_vocab_size = self.qb_embeddings.shape[0]
        qb_we_dimension = self.qb_embeddings.shape[1]

        # Keras Embeddings only supports 2 dimensional input so we have to do reshape ninjitsu to make this work
        wiki_input = Input((self.n_memories, self.wiki_max_len,), name='wiki_input')
        qb_input = Input((self.qb_max_len,), name='wiki_input')

        wiki_embeddings = Embedding(wiki_vocab_size, wiki_we_dimension, weights=[self.wiki_embeddings], mask_zero=True)
        qb_embeddings = Embedding(qb_vocab_size, qb_we_dimension, weights=[self.qb_embeddings], mask_zero=True)

        # encoders

        # Wikipedia sentence encoder used to search memory
        wiki_m_encoder = Sequential()
        wiki_m_encoder.add(Reshape((self.n_memories * self.wiki_max_len,)))
        wiki_m_encoder.add(wiki_embeddings)
        wiki_m_encoder.add(Reshape((self.n_memories, self.wiki_max_len, wiki_we_dimension)))
        wiki_m_encoder.add(AverageWords())
        wiki_input_encoded_m = wiki_m_encoder(wiki_input)

        # Wikipedia sentence encoder for retrieved memories
        wiki_c_encoder = Sequential()
        wiki_c_encoder.add(Reshape((self.n_memories * self.wiki_max_len,)))
        wiki_c_encoder.add(wiki_embeddings)
        wiki_c_encoder.add(Reshape((self.n_memories, self.wiki_max_len, wiki_we_dimension)))
        wiki_c_encoder.add(AverageWords())
        wiki_input_encoded_c = wiki_c_encoder(wiki_input)

        # Quiz Bowl question encoder
        qb_encoder = Sequential()
        qb_encoder.add(qb_embeddings)
        qb_encoder.add(WordDropout(self.word_dropout_rate))
        qb_encoder.add(AverageWords())
        qb_input_encoded = qb_encoder(qb_input)

        # Compute the attention based on match between memory addresses and question input
        match_probability = dot([wiki_input_encoded_m, qb_input_encoded], 1)
        match_probability = Activation('softmax')(match_probability)

        memories = multiply([match_probability, wiki_input_encoded_c])
        memories_and_question = add([memories, qb_input_encoded])

        actions = Dense(self.n_classes)(memories_and_question)
        actions = BatchNormalization()(actions)
        actions = Dropout(self.nn_dropout_rate)(actions)
        actions = Activation('softmax')(actions)

        adam = Adam()
        model = Model(inputs=[qb_input, wiki_input], outputs=actions)
        model.compile(
            loss=sparse_categorical_crossentropy, optimizer=adam,
            metrics=['sparse_categorical_accuracy']
        )
        return model

    def train(self, training_data: TrainingData) -> None:
        log.info('Preprocessing training data...')
        x_train, y_train, _, x_test, y_test, _, qb_vocab, class_to_i, i_to_class = preprocess_dataset(training_data)
        self.class_to_i = class_to_i
        self.i_to_class = i_to_class
        self.qb_vocab = qb_vocab

        log.info('Creating qb embeddings...')
        qb_embeddings, qb_embedding_lookup = qb_load_embeddings(
            vocab=qb_vocab, expand_glove=self.expand_we, mask_zero=True
        )
        self.qb_embeddings = qb_embeddings
        self.qb_embedding_lookup = qb_embedding_lookup

        log.info('Converting qb dataset to embeddings...')
        x_train = [nn.convert_text_to_embeddings_indices(q, qb_embedding_lookup) for q in x_train]
        x_test = [nn.convert_text_to_embeddings_indices(q, qb_embedding_lookup) for q in x_test]
        self.n_classes = nn.compute_n_classes(training_data[1])
        self.qb_max_len = nn.compute_max_len(training_data)
        x_train = pad_sequences(x_train, maxlen=self.qb_max_len, value=0, padding='post', truncating='post')
        x_test = pad_sequences(x_test, maxlen=self.qb_max_len, value=0, padding='post', truncating='post')

        log.info('Collecting Wikipedia data...')
        classes = set(i_to_class)
        wiki_sentences = build_wikipedia_sentences(classes, self.n_wiki_sentences)

        log.info('Building Memory Index...')
        MemoryIndex.build(wiki_sentences)

        log.info('Fetching most relevant {} memories per question in train and test...'.format(self.n_memories))
        log.info('Fetching train memories...')
        train_memories = MemoryIndex.search_parallel(x_train, self.n_memories)
        log.info('Fetching test memories...')
        test_memories = MemoryIndex.search_parallel(x_test, self.n_memories)

        tokenized_train_memories, wiki_vocab = preprocess_wikipedia(train_memories, True)
        tokenized_test_memories, _ = preprocess_wikipedia(test_memories, False)

        log.info('Creating wiki embeddings...')
        wiki_embeddings, wiki_embedding_lookup = wiki_load_embeddings(
            vocab=wiki_vocab, expand_glove=self.expand_we, mask_zero=True
        )

        wiki_max_len = 0
        for i in range(len(tokenized_train_memories)):
            memories = tokenized_train_memories[i]
            we_memories = [nn.convert_text_to_embeddings_indices(m, wiki_embedding_lookup) for m in memories]
            if len(we_memories) != 0:
                wiki_max_len = max(wiki_max_len, max(len(we_m) for we_m in we_memories))
            tokenized_train_memories[i] = we_memories

        self.wiki_max_len = wiki_max_len
        tokenized_train_memories = pad_sequences(tokenized_train_memories,
                                                 maxlen=self.wiki_max_len, value=0, padding='post', truncating='post')

        for i in range(len(tokenized_test_memories)):
            memories = tokenized_test_memories[i]
            we_memories = [nn.convert_text_to_embeddings_indices(m, wiki_embedding_lookup) for m in memories]
            tokenized_test_memories[i] = we_memories

        tokenized_test_memories = pad_sequences(tokenized_test_memories,
                                                maxlen=self.wiki_max_len, value=0, padding='post', truncating='post')

        log.info('Converting wiki dataset to embeddings...')

        log.info('Building keras model...')
        self.model = self.build_model()

        log.info('Training model...')
        callbacks = [
            TensorBoard(),
            EarlyStopping(patience=self.max_patience, monitor='val_sparse_categorical_accuracy'),
            ModelCheckpoint(
                safe_path(MEM_MODEL_TMP_TARGET),
                save_best_only=True,
                monitor='val_sparse_categorical_accuracy'
            )
        ]
        history = self.model.fit(
            x_train, y_train,
            validation_data=(x_test, y_test),
            batch_size=self.batch_size, epochs=self.max_n_epochs,
            callbacks=callbacks, verbose=2
        )
        log.info('Done training')
        log.info('Printing model training history...')
        log.info(history.history)

    def guess(self, questions: List[QuestionText], max_n_guesses: Optional[int]) -> List[List[Tuple[Answer, float]]]:
        log.info('Generating {} guesses for each of {} questions'.format(max_n_guesses, len(questions)))
        x_test = [nn.convert_text_to_embeddings_indices(tokenize_question(q), self.embedding_lookup) for q in questions]
        x_test = np.array(nn.tf_format(x_test, self.qb_max_len, 0))
        class_probabilities = self.model.predict_proba(x_test, batch_size=self.batch_size)
        guesses = []
        for row in class_probabilities:
            sorted_labels = np.argsort(-row)[:max_n_guesses]
            sorted_guesses = [self.i_to_class[i] for i in sorted_labels]
            sorted_scores = np.copy(row[sorted_labels])
            guesses.append(list(zip(sorted_guesses, sorted_scores)))
        return guesses

    def save(self, directory: str) -> None:
        shutil.copyfile(MEM_MODEL_TMP_TARGET, os.path.join(directory, MEM_MODEL_TARGET))
        with safe_open(os.path.join(directory, MEM_PARAMS_TARGET), 'wb') as f:
            pickle.dump(self.dump_parameters(), f)

    @classmethod
    def load(cls, directory: str):
        guesser = MemNNGuesser()
        guesser.model = load_model(
            os.path.join(directory, MEM_MODEL_TARGET),
            custom_objects={
                'GlobalAveragePooling1DMasked': AverageWords,
                'WordDropout': WordDropout
            }
        )
        with open(os.path.join(directory, MEM_PARAMS_TARGET), 'rb') as f:
            params = pickle.load(f)
            guesser.load_parameters(params)

        return guesser
