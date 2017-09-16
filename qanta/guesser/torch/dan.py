from typing import List, Tuple, Optional
import time

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from qanta import logging
from qanta.guesser.abstract import AbstractGuesser
from qanta.datasets.abstract import TrainingData, Answer, QuestionText
from qanta.preprocess import preprocess_dataset
from qanta.guesser.nn import create_load_embeddings_function, convert_text_to_embeddings_indices, compute_n_classes


log = logging.get(__name__)


PTDAN_WE_TMP = '/tmp/qanta/deep/pt_dan_we.pickle'
PTDAN_WE = 'pt_dan_we.pickle'
load_embeddings = create_load_embeddings_function(PTDAN_WE_TMP, PTDAN_WE, log)


class DanGuesser(AbstractGuesser):
    def __init__(self, max_epochs=100, batch_size=128):
        super(DanGuesser, self).__init__()
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.class_to_i = None
        self.i_to_class = None
        self.vocab = None
        self.embeddings = None
        self.embedding_lookup = None
        self.n_classes = None
        self.criterion = nn.CrossEntropyLoss()

    def guess(self, questions: List[QuestionText], max_n_guesses: Optional[int]) -> List[List[Tuple[Answer, float]]]:
        pass

    def save(self, directory: str) -> None:
        pass

    def train(self, training_data: TrainingData) -> None:
        x_train_text, y_train, x_test_text, y_test, vocab, class_to_i, i_to_class = preprocess_dataset(
            training_data
        )

        self.class_to_i = class_to_i
        self.i_to_class = i_to_class
        self.vocab = vocab

        embeddings, embedding_lookup = load_embeddings(vocab=vocab, expand_glove=True)
        self.embeddings = embeddings
        self.embedding_lookup = embedding_lookup

        x_train = np.array([convert_text_to_embeddings_indices(q, embedding_lookup) for q in x_train_text])
        y_train = np.array(y_train)

        x_test = np.array([convert_text_to_embeddings_indices(q, embedding_lookup) for q in x_test_text])
        y_test = np.array(y_test)

        self.n_classes = compute_n_classes(training_data[1])

        model = DanModel(embeddings.shape[0], self.n_classes)
        model.train()
        if torch.cuda.is_available():
            model.cuda()

        optimizer = torch.optim.Adam(model.parameters())

        epoch_losses = []
        start_time = time.time()
        for epoch in range(self.max_epochs):
            log.info('Starting epoch: {}'.format(epoch))
            epoch_loss = self.run_epoch(model, optimizer, x_train, y_train, x_test, y_test)
            epoch_losses.append(epoch_loss)
            log.info('Loss: {}'.format(epoch_loss))

        end_time = time.time()

    def run_epoch(self, model, optimizer, x_train, y_train, x_test, y_test):
        n_examples = x_train.shape[0]
        random_order = np.random.permutation(n_examples)
        x_train = x_train[random_order]
        y_train = y_train[random_order]

        n_batches = n_examples // self.batch_size
        epoch_loss = 0
        epoch_start = time.time()
        batch_losses = []
        batch_accuracies = []

        for batch in range(n_batches):
            x_batch = x_train[batch * self.batch_size:(batch + 1) * self.batch_size]
            y_batch = y_train[batch * self.batch_size:(batch + 1) * self.batch_size]
            flat_x_batch = []
            for r in x_batch:
                flat_x_batch.extend(r)
            flat_x_batch = np.array(flat_x_batch)
            x_lengths = [len(r) for r in x_batch]

            t_x_batch = Variable(torch.from_numpy(flat_x_batch).long().cuda())
            t_offsets = Variable(torch.from_numpy(np.cumsum([0] + x_lengths[:-1])).long().cuda())
            t_y_batch = Variable(torch.from_numpy(y_batch).long().cuda())

            model.zero_grad()
            logits = model(t_x_batch, t_offsets)
            _, preds = torch.max(logits, 1)
            accuracy = torch.mean(torch.eq(preds, t_y_batch).float()).data[0]
            batch_accuracies.append(accuracy)
            batch_loss = self.criterion(logits, t_y_batch)
            batch_losses.append(batch_loss.data[0])
            batch_loss.backward()
            optimizer.step()

        epoch_end = time.time()
        return np.mean(batch_losses), np.mean(batch_accuracies), epoch_end - epoch_start

    @classmethod
    def load(cls, directory: str):
        pass

    @classmethod
    def targets(cls) -> List[str]:
        pass


class DanModel(nn.Module):
    def __init__(self, vocab_size, n_classes,
                 embedding_dim=300,
                 dropout_prob=.5, word_dropout_prob=.5,
                 n_hidden_layers=1, n_hidden_units=1000, non_linearity='relu',
                 init_scale=.1):
        super(DanModel, self).__init__()
        self.n_hidden_layers = 1
        self.non_linearity = non_linearity
        self.n_hidden_units = n_hidden_units
        self.dropout_prob = dropout_prob
        self.word_dropout_prob = word_dropout_prob
        self.vocab_size = vocab_size
        self.n_classes = n_classes
        self.embedding_dim = embedding_dim
        self.init_scale = init_scale

        self.dropout = nn.Dropout(dropout_prob)
        self.word_dropout = nn.Dropout2d(word_dropout_prob)
        self.embeddings = nn.EmbeddingBag(vocab_size, embedding_dim)

        layers = []
        for i in range(n_hidden_layers):
            if i == 0:
                input_dim = embedding_dim
            else:
                input_dim = n_hidden_units

            layers.extend([
                nn.Linear(input_dim, n_hidden_units),
                nn.Dropout(dropout_prob),
                nn.ReLU()
            ])

        layers.extend([
            nn.Linear(n_hidden_units, n_classes),
            nn.Dropout(dropout_prob)
        ])
        self.layers = nn.Sequential(*layers)

        self.init_weights()

    def init_weights(self, initial_embeddings=None):
        if initial_embeddings is not None:
            self.embeddings.weight = nn.Parameter(torch.from_numpy(initial_embeddings).float())

    def forward(self, input_: Variable, offsets: Variable):
        avg_embeddings = self.dropout(self.embeddings(input_.view(-1), offsets))
        return self.layers(avg_embeddings)
