from typing import List, Tuple, Optional

import torch.nn as nn
from torch.autograd import Variable

from qanta.guesser.abstract import AbstractGuesser
from qanta.datasets.abstract import TrainingData, Answer, QuestionText


class DanGuesser(AbstractGuesser):
    def guess(self, questions: List[QuestionText], max_n_guesses: Optional[int]) -> List[List[Tuple[Answer, float]]]:
        pass

    def save(self, directory: str) -> None:
        pass

    def train(self, training_data: TrainingData) -> None:
        pass

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
                 n_hidden_layers=1, n_hidden_units=1000, non_linearity='relu'):
        super(DanModel, self).__init__()
        self.n_hidden_layers = 1
        self.non_linearity = non_linearity
        self.n_hidden_units = n_hidden_units
        self.dropout_prob = dropout_prob
        self.word_dropout_prob = word_dropout_prob
        self.vocab_size = vocab_size
        self.n_classes = n_classes
        self.embedding_dim = embedding_dim

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
            nn.Dropout(dropout_prob),
            nn.LogSoftmax()
        ])
        self.layers = nn.Sequential(*layers)
