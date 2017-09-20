import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

from qanta import logging
from qanta.guesser.abstract import AbstractGuesser
from qanta.datasets.abstract import TrainingData, Answer, QuestionText
from qanta.preprocess import preprocess_dataset, tokenize_question
from qanta.guesser.nn import create_load_embeddings_function, convert_text_to_embeddings_indices, compute_n_classes
from qanta import manager
from qanta.guesser.torch.util import create_save_model


log = logging.get(__name__)

PT_RNN_WE_TMP = '/tmp/qanta/deep/pt_rnn_we.pickle'
PT_RNN_WE = 'pt_rnn_we.pickle'
load_embeddings = create_load_embeddings_function(PT_RNN_WE_TMP, PT_RNN_WE, log)


def repackage_hidden(hidden):
    if type(hidden) == Variable:
        return Variable(hidden.data)
    else:
        return tuple(repackage_hidden(v) for v in hidden)


class RnnModel(nn.Module):
    def __init__(self, vocab_size, n_classes, embedding_dim=300, dropout_prob=.3, recurrent_dropout_prob=.3,
                 n_hidden_layers=1, n_hidden_units=1000):
        super(RnnModel, self).__init__()
        self.vocab_size = vocab_size
        self.n_classes = n_classes
        self.embedding_dim = embedding_dim
        self.dropout_prob = dropout_prob
        self.recurrent_dropout_prob = recurrent_dropout_prob
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_units = n_hidden_units

        self.dropout = nn.Dropout(dropout_prob)
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, n_hidden_units, n_hidden_layers, dropout=recurrent_dropout_prob)
        self.classification_layer = nn.Sequential(
            nn.Linear(n_hidden_units, n_classes),
            nn.BatchNorm1d(n_classes),
            nn.Dropout(dropout_prob)
        )

    def init_weights(self, embeddings=None):
        pass

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (
            Variable(weight.new(self.n_hidden_layers, batch_size, self.n_hidden_units).zero_()),
            Variable(weight.new(self.n_hidden_layers, batch_size, self.n_hidden_units).zero_())
        )

    def forward(self, input_: Variable, hidden):
        embeddings = self.dropout(self.embeddings(input_))
        output, hidden = self.rnn(embeddings, hidden)
        return self.classification_layer(output), hidden