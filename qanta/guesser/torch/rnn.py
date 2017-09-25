from typing import List, Optional
import time
import pickle
import os
import shutil

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
from qanta.manager import (
    BaseLogger, TerminateOnNaN, Tensorboard,
    EarlyStopping, ModelCheckpoint, MaxEpochStopping, TrainingManager
)
from qanta.guesser.torch.util import create_save_model


log = logging.get(__name__)

PT_RNN_WE_TMP = '/tmp/qanta/deep/pt_rnn_we.pickle'
PT_RNN_WE = 'pt_rnn_we.pickle'
load_embeddings = create_load_embeddings_function(PT_RNN_WE_TMP, PT_RNN_WE, log)


def repackage_hidden(hidden, reset=False):
    if type(hidden) == Variable:
        if reset:
            return Variable(hidden.data.zero_())
        else:
            return Variable(hidden.data)
    else:
        return tuple(repackage_hidden(v) for v in hidden)


def create_batch(x_array, y_array):
    lengths = np.array([len(r) for r in x_array])
    max_length = np.max(lengths)
    length_sort = np.argsort(-lengths)
    x_batch = x_array[length_sort]
    y_batch = y_array[length_sort]
    lengths = lengths[length_sort]

    x_batch_padded = []
    for r in x_batch:
        pad_r = list(r)
        while len(pad_r) < max_length:
            pad_r.append(0)
        x_batch_padded.append(pad_r)
    x_batch_padded = np.array(x_batch_padded)

    x_batch_padded = torch.from_numpy(x_batch_padded).long().cuda()
    y_batch = torch.from_numpy(y_batch).long().cuda()

    return x_batch_padded, lengths, y_batch, length_sort


def batchify(batch_size, x_array, y_array, truncate=True, shuffle=True):
    n_examples = x_array.shape[0]
    n_batches = n_examples // batch_size
    if shuffle:
        random_order = np.random.permutation(n_examples)
        x_array = x_array[random_order]
        y_array = y_array[random_order]

    t_x_batches = []
    length_batches = []
    t_y_batches = []
    sort_batches = []

    for b in range(n_batches):
        x_batch = x_array[b * batch_size:(b + 1) * batch_size]
        y_batch = y_array[b * batch_size:(b + 1) * batch_size]
        x_batch, lengths, y_batch, sort = create_batch(x_batch, y_batch)

        t_x_batches.append(x_batch)
        length_batches.append(lengths)
        t_y_batches.append(y_batch)
        sort_batches.append(sort)

    if (not truncate) and (batch_size * n_batches < n_examples):
        x_batch = x_array[n_batches * batch_size:]
        y_batch = y_array[n_batches * batch_size:]

        x_batch, lengths, y_batch, sort = create_batch(x_batch, y_batch)

        t_x_batches.append(x_batch)
        length_batches.append(lengths)
        t_y_batches.append(y_batch)
        sort_batches.append(sort)

    return n_batches, t_x_batches, length_batches, t_y_batches, sort_batches


class RnnGuesser(AbstractGuesser):
    def __init__(self, max_epochs=100, batch_size=256, learning_rate=.001, max_grad_norm=5):
        super(RnnGuesser, self).__init__()
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.class_to_i = None
        self.i_to_class = None
        self.vocab = None
        self.embeddings = None
        self.embedding_lookup = None
        self.n_classes = None
        self.model = None
        self.criterion = None
        self.optimizer = None

    def guess(self,
              questions: List[QuestionText],
              max_n_guesses: Optional[int]):
        x_test = [convert_text_to_embeddings_indices(
            tokenize_question(q), self.embedding_lookup)
            for q in questions
        ]
        for r in x_test:
            if len(r) == 0:
                log.warn('Found an empty question, adding an UNK token to it so that NaNs do not occur')
                r.append(self.embedding_lookup['UNK'])
        x_test = np.array(x_test)
        y_test = np.zeros(len(x_test))

        _, t_x_batches, lengths, t_y_batches, sort_batches = batchify(
            self.batch_size, x_test, y_test, truncate=False, shuffle=False)

        self.model.eval()
        self.model.cuda()
        guesses = []
        hidden = self.model.init_hidden(self.batch_size)
        for b in range(len(t_x_batches)):
            t_x = Variable(t_x_batches[b], volatile=True)
            length_batch = lengths[b]
            sort = sort_batches[b]

            if len(length_batch) != self.batch_size:
                # This could happen for the last batch which is shorter than batch_size
                hidden = self.model.init_hidden(len(length_batch))
            else:
                hidden = repackage_hidden(hidden, reset=True)

            out, hidden = self.model(t_x, length_batch, hidden)
            probs = F.softmax(out)
            scores, preds = torch.max(probs, 1)
            scores = scores.data.cpu().numpy()[np.argsort(sort)]
            preds = preds.data.cpu().numpy()[np.argsort(sort)]
            for p, s in zip(preds, scores):
                guesses.append([(self.i_to_class[p], s)])

        return guesses

    def train(self, training_data: TrainingData):
        x_train_text, y_train, x_test_text, y_test, vocab, class_to_i, i_to_class = preprocess_dataset(
            training_data
        )

        self.class_to_i = class_to_i
        self.i_to_class = i_to_class
        self.vocab = vocab

        embeddings, embedding_lookup = load_embeddings(vocab=vocab, expand_glove=True, mask_zero=True)
        self.embeddings = embeddings
        self.embedding_lookup = embedding_lookup

        x_train = [convert_text_to_embeddings_indices(q, embedding_lookup) for q in x_train_text]
        for r in x_train:
            if len(r) == 0:
                r.append(embedding_lookup['UNK'])
        x_train = np.array(x_train)
        y_train = np.array(y_train)

        x_test = [convert_text_to_embeddings_indices(q, embedding_lookup) for q in x_test_text]
        for r in x_test:
            if len(r) == 0:
                r.append(embedding_lookup['UNK'])
        x_test = np.array(x_test)
        y_test = np.array(y_test)

        self.n_classes = compute_n_classes(training_data[1])

        n_batches_train, t_x_train, lengths_train, t_y_train, _ = batchify(
            self.batch_size, x_train, y_train, truncate=True)
        n_batches_test, t_x_test, lengths_test, t_y_test, _ = batchify(
            self.batch_size, x_test, y_test, truncate=False)

        self.model = RnnModel(embeddings.shape[0], self.n_classes)
        self.model.init_weights(embeddings=embeddings)
        self.model.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        manager = TrainingManager([
            BaseLogger(log_func=log.info), TerminateOnNaN(),
            EarlyStopping(monitor='test_acc', patience=10, verbose=1), MaxEpochStopping(100),
            ModelCheckpoint(create_save_model(self.model), '/tmp/rnn.pt', monitor='test_acc'),
            Tensorboard('rnn', log_dir='tb-logs')
        ])

        log.info('Starting training...')
        while True:
            self.model.train()
            train_acc, train_loss, train_time = self.run_epoch(
                n_batches_train,
                t_x_train, lengths_train, t_y_train, evaluate=False
            )

            self.model.eval()
            test_acc, test_loss, test_time = self.run_epoch(
                n_batches_test,
                t_x_test, lengths_test, t_y_test, evaluate=True
            )

            stop_training, reasons = manager.instruct(
                train_time, train_loss, train_acc,
                test_time, test_loss, test_acc
            )

            if stop_training:
                log.info(' '.join(reasons))
                break

        log.info('Done training')

    def run_epoch(self, n_batches, t_x_array, lengths_list, t_y_array, evaluate=False):
        if evaluate:
            batch_order = range(n_batches)
        else:
            batch_order = np.random.permutation(n_batches)

        batch_accuracies = []
        batch_losses = []
        epoch_start = time.time()
        hidden = self.model.init_hidden(self.batch_size)
        for batch in batch_order:
            t_x_batch = Variable(t_x_array[batch], volatile=evaluate)
            length_batch = lengths_list[batch]
            t_y_batch = Variable(t_y_array[batch], volatile=evaluate)

            self.model.zero_grad()
            hidden = repackage_hidden(hidden, reset=True)
            out, hidden = self.model(t_x_batch, length_batch, hidden)
            _, preds = torch.max(out, 1)
            accuracy = torch.mean(torch.eq(preds, t_y_batch).float()).data[0]
            batch_loss = self.criterion(out, t_y_batch)
            if not evaluate:
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm(self.model.parameters(), 5)
                self.optimizer.step()

            batch_accuracies.append(accuracy)
            batch_losses.append(batch_loss.data[0])

        epoch_end = time.time()

        return np.mean(batch_accuracies), np.mean(batch_losses), epoch_end - epoch_start

    def save(self, directory: str):
        shutil.copyfile('/tmp/rnn.pt', os.path.join(directory, 'rnn.pt'))
        with open(os.path.join(directory, 'rnn.pickle'), 'wb') as f:
            pickle.dump({
                'vocab': self.vocab,
                'class_to_i': self.class_to_i,
                'i_to_class': self.i_to_class,
                'embeddings': self.embeddings,
                'embedding_lookup': self.embedding_lookup,
                'n_classes': self.n_classes,
                'max_epochs': self.max_epochs,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate
            }, f)

    @classmethod
    def load(cls, directory: str):
        with open(os.path.join(directory, 'rnn.pickle'), 'rb') as f:
            params = pickle.load(f)

        guesser = RnnGuesser()
        guesser.vocab = params['vocab']
        guesser.class_to_i = params['class_to_i']
        guesser.i_to_class = params['i_to_class']
        guesser.embeddings = params['embeddings']
        guesser.embedding_lookup = params['embedding_lookup']
        guesser.n_classes = params['n_classes']
        guesser.max_epochs = params['max_epochs']
        guesser.batch_size = params['batch_size']
        guesser.learning_rate = params['learning_rate']
        guesser.model = torch.load(os.path.join(directory, 'rnn.pt'))
        return  guesser

    @classmethod
    def targets(cls):
        return ['rnn.pickle', 'rnn.pt']


class RnnModel(nn.Module):
    def __init__(self, vocab_size, n_classes, embedding_dim=300, dropout_prob=0, recurrent_dropout_prob=0,
                 n_hidden_layers=1, n_hidden_units=1000, bidirectional=True, rnn_type='gru',
                 rnn_output='max_pool'):
        super(RnnModel, self).__init__()
        self.vocab_size = vocab_size
        self.n_classes = n_classes
        self.embedding_dim = embedding_dim
        self.dropout_prob = dropout_prob
        self.recurrent_dropout_prob = recurrent_dropout_prob
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_units = n_hidden_units
        self.bidirectional = bidirectional
        self.rnn_output = rnn_output

        #self.dropout = nn.Dropout(dropout_prob)
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn_type = rnn_type
        if rnn_type == 'lstm':
            rnn_layer = nn.LSTM
        elif rnn_type == 'gru':
            rnn_layer = nn.GRU
        else:
            raise ValueError('Unrecognized rnn layer type')
        self.rnn = rnn_layer(embedding_dim, n_hidden_units, n_hidden_layers,
                           dropout=recurrent_dropout_prob, batch_first=True, bidirectional=bidirectional)
        self.num_directions = int(bidirectional) + 1
        self.classification_layer = nn.Sequential(
            nn.Linear(n_hidden_units * self.num_directions * self.n_hidden_layers, n_classes)
            #nn.BatchNorm1d(n_classes),
            #nn.Dropout(dropout_prob)
        )

    def init_weights(self, embeddings=None):
        if embeddings is not None:
            self.embeddings.weight = nn.Parameter(torch.from_numpy(embeddings).float())

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        if self.rnn_type == 'lstm':
            return (
                Variable(weight.new(self.n_hidden_layers * self.num_directions, batch_size, self.n_hidden_units).zero_()),
                Variable(weight.new(self.n_hidden_layers * self.num_directions, batch_size, self.n_hidden_units).zero_())
            )
        else:
            return Variable(weight.new(self.n_hidden_layers * self.num_directions, batch_size, self.n_hidden_units).zero_())

    def forward(self, input_: Variable, lengths, hidden):
        #embeddings = self.dropout(self.embeddings(input_))
        embeddings = self.embeddings(input_)
        packed_input = nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=True)

        output, hidden = self.rnn(packed_input, hidden)

        #padded_sequence, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        #positions = torch.LongTensor(lengths).cuda() - 1
        #row_indexer = torch.arange(0, padded_sequence.data.shape[0]).long().cuda()
        #last_out = padded_sequence[row_indexer, positions]
        #return self.classification_layer(last_out), hidden
        if self.rnn_output == 'last_hidden':
            if type(hidden) == tuple:
                final_hidden = hidden[0]
            else:
                final_hidden = hidden

            h_reshaped = final_hidden.transpose(0, 1).contiguous().view(input_.data.shape[0], -1)

            return self.classification_layer(h_reshaped), hidden
        elif self.rnn_output == 'max_pool':
            idx = np.cumsum(np.insert(lengths, 0, 0))
            pooled = []
            for i in range(len(lengths)):
                pooled.append(output.data[idx[i]:idx[i + 1]].max(0)[0].view(-1))
                #pooled.append(output.data[idx[i]:idx[i + 1]].mean(0).view(-1))
            pooled = torch.cat(pooled).view(len(lengths), -1)
            return self.classification_layer(pooled), hidden
        else:
            raise ValueError('Unrecognized rnn_output option')

