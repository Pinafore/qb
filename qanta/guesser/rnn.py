import re
import os
import shutil
import time
import pickle
from typing import List, Optional, Dict

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.optim import Adam, lr_scheduler

from torchtext.data.field import Field
from torchtext.data.iterator import Iterator

from qanta import qlogging
from qanta.torch.dataset import QuizBowl
from qanta.config import conf
from qanta.guesser.abstract import AbstractGuesser
from qanta.datasets.abstract import QuestionText
from qanta.torch import (
    BaseLogger, TerminateOnNaN, EarlyStopping, ModelCheckpoint,
    MaxEpochStopping, TrainingManager
)
from qanta.torch.nn import WeightDrop, LockedDropout


log = qlogging.get(__name__)


PT_RNN_WE_TMP = '/tmp/qanta/deep/pt_rnn_we.pickle'
PT_RNN_WE = 'pt_rnn_we.pickle'
CUDA = torch.cuda.is_available()


def create_save_model(model):
    def save_model(path):
        torch.save(model.state_dict(), path)
    return save_model



qb_patterns = {
    '\n',
    ', for 10 points,',
    ', for ten points,',
    '--for 10 points--',
    'for 10 points, ',
    'for 10 points--',
    'for ten points, ',
    'for 10 points ',
    'for ten points ',
    ', ftp,'
    'ftp,',
    'ftp',
    '(*)'
}
re_pattern = '|'.join([re.escape(p) for p in qb_patterns])
re_pattern += r'|\[.*?\]|\(.*?\)'


class DanEncoder(nn.Module):
    def __init__(self, embedding_dim, n_hidden_layers, n_hidden_units, dropout_prob):
        super(DanEncoder, self).__init__()
        encoder_layers = []
        for i in range(n_hidden_layers):
            if i == 0:
                input_dim = embedding_dim
            else:
                input_dim = n_hidden_units

            encoder_layers.extend([
                nn.Linear(input_dim, n_hidden_units),
                nn.BatchNorm1d(n_hidden_units),
                nn.ELU(),
                nn.Dropout(dropout_prob),
            ])
        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, x_array):
        return self.encoder(x_array)


class Model(nn.Module):
    def __init__(self, text_field: Field, n_classes,
                 init_embeddings=True, emb_dim=300,
                 n_hidden_units=1000, n_hidden_layers=1, nn_dropout=.265, sm_dropout=.158):
        super(Model, self).__init__()
        vocab = text_field.vocab
        self.vocab_size = len(vocab)
        self.emb_dim = emb_dim
        self.n_classes = n_classes
        self.n_hidden_units = n_hidden_units
        self.n_hidden_layers = n_hidden_layers
        self.nn_dropout = nn_dropout
        self.sm_dropout = sm_dropout

        self.dropout = nn.Dropout(nn_dropout)
        self.embeddings = nn.Embedding(self.vocab_size, emb_dim, padding_idx=vocab.stoi[text_field.pad_token])
        if init_embeddings:
            mean_emb = vocab.vectors.mean(0)
            vocab.vectors[vocab.stoi[text_field.unk_token]] = mean_emb
            self.embeddings.weight.data = vocab.vectors.cuda()
        self.encoder = DanEncoder(emb_dim, self.n_hidden_layers, self.n_hidden_units, self.nn_dropout)
        self.classifier = nn.Sequential(
            nn.Linear(self.n_hidden_units, n_classes),
            nn.BatchNorm1d(n_classes),
            nn.Dropout(self.sm_dropout)
        )

    def forward(self, input_: Variable, lengths):
        if not isinstance(lengths, Variable):
            lengths = Variable(lengths.float(), volatile=not self.training)
        encoded = self.embeddings(input_)
        encoded = encoded.sum(1) / lengths.float().view(input_.size()[0], -1)
        encoded = self.encoder(self.dropout(encoded))
        return self.classifier(encoded)



class RnnGuesser(AbstractGuesser):
    def __init__(self):
        super(RnnGuesser, self).__init__()
        guesser_conf = conf['guessers']['RNN']
        self.gradient_clip = .25
        self.n_hidden_units = 1000
        self.n_hidden_layers = 1
        self.lr = .001
        self.nn_dropout = .265
        self.sm_dropout = .158
        self.batch_size = 1024
        self.emb_dim = 300

        self.page_field: Optional[Field] = None
        self.qnum_field: Optional[Field] = None
        self.text_field: Optional[Field] = None
        self.n_classes = None

        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scheduler = None

    @property
    def ans_to_i(self):
        return self.page_field.vocab.stoi

    @property
    def i_to_ans(self):
        return self.page_field.vocab.itos

    def parameters(self):
        return {}

    def train(self, training_data):
        log.info('Loading Quiz Bowl dataset')
        train_iter, val_iter, dev_iter = QuizBowl.iters(batch_size=self.batch_size)
        log.info(f'N Train={len(train_iter.dataset.examples)}')
        log.info(f'N Test={len(val_iter.dataset.examples)}')
        fields: Dict[str, Field] = train_iter.dataset.fields
        self.page_field = fields['page']
        self.n_classes = len(self.ans_to_i)
        self.qnum_field = fields['qnum']
        self.text_field = fields['text']
        log.info(f'Vocab={len(self.text_field.vocab)}')

        log.info('Initializing Model')
        self.model = Model(
            self.text_field, self.n_classes, emb_dim=self.emb_dim,
            n_hidden_units=self.n_hidden_units, n_hidden_layers=self.n_hidden_layers,
            nn_dropout=self.nn_dropout, sm_dropout=self.sm_dropout,
        )
        if CUDA:
            self.model = self.model.cuda()
        log.info(f'Model:\n{self.model}')
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, verbose=True, mode='max')

        manager = TrainingManager([
            BaseLogger(log_func=log.info), TerminateOnNaN(), EarlyStopping(monitor='test_acc', patience=10, verbose=1),
            MaxEpochStopping(100), ModelCheckpoint(create_save_model(self.model), '/tmp/rnn.pt', monitor='test_acc')
        ])

        log.info('Starting training')
        while True:
            self.model.train()
            train_acc, train_loss, train_time = self.run_epoch(train_iter)

            self.model.eval()
            test_acc, test_loss, test_time = self.run_epoch(val_iter)

            stop_training, reasons = manager.instruct(
                train_time, train_loss, train_acc,
                test_time, test_loss, test_acc
            )

            if stop_training:
                log.info(' '.join(reasons))
                break
            else:
                self.scheduler.step(test_acc)

    def run_epoch(self, iterator: Iterator):
        is_train = iterator.train
        batch_accuracies = []
        batch_losses = []
        epoch_start = time.time()
        for batch in iterator:
            text, lengths = batch.text
            page = batch.page

            if is_train:
                self.model.zero_grad()

            out = self.model(text, lengths)
            _, preds = torch.max(out, 1)
            accuracy = torch.mean(torch.eq(preds, page).float()).data[0]
            batch_loss = self.criterion(out, page)
            if is_train:
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm(self.model.parameters(), self.gradient_clip)
                self.optimizer.step()

            batch_accuracies.append(accuracy)
            batch_losses.append(batch_loss.data[0])

        epoch_end = time.time()

        return np.mean(batch_accuracies), np.mean(batch_losses), epoch_end - epoch_start

    def guess(self, questions: List[QuestionText], max_n_guesses: Optional[int]):
        examples = [self.text_field.preprocess(q) for q in questions]
        text, lengths = self.text_field.process(examples, None, False)
        guesses = []
        out = self.model(text, lengths)
        probs = F.softmax(out)
        scores, preds = torch.max(probs, 1)
        scores = scores.data.cpu().numpy()
        preds = preds.data.cpu().numpy()

        for p, s in zip(preds, scores):
            guesses.append([(self.i_to_ans[p], s)])

        return guesses


    def save(self, directory: str):
        shutil.copyfile('/tmp/rnn.pt', os.path.join(directory, 'rnn.pt'))
        with open(os.path.join(directory, 'rnn.pkl'), 'wb') as f:
            pickle.dump({
                'page_field': self.page_field,
                'text_field': self.text_field,
                'qnum_field': self.qnum_field,
                'n_classes': self.n_classes,
                'emb_dim': self.emb_dim,
                'gradient_clip': self.gradient_clip,
                'n_hidden_units': self.n_hidden_units,
                'n_hidden_layers': self.n_hidden_layers,
                'lr': self.lr,
                'nn_dropout': self.nn_dropout,
                'sm_dropout': self.sm_dropout,
                'batch_size': self.batch_size
            }, f)

    @classmethod
    def load(cls, directory: str):
        with open(os.path.join(directory, 'rnn.pkl'), 'rb') as f:
            params = pickle.load(f)

        guesser = RnnGuesser()
        guesser.page_field = params['page_field']
        guesser.text_field = params['text_field']
        guesser.qnum_field = params['qnum_field']
        guesser.n_classes = params['n_classes']
        guesser.emb_dim = params['emb_dim']
        guesser.gradient_clip = params['gradient_clip']
        guesser.n_hidden_units = params['n_hidden_units']
        guesser.n_hidden_layers = params['n_hidden_layers']
        guesser.lr = params['lr']
        guesser.nn_dropout = params['nn_dropout']
        guesser.sm_dropout = params['sm_dropout']
        guesser.model = Model(
            guesser.text_field, guesser.n_classes,
            init_embeddings=False, emb_dim=guesser.emb_dim
        )
        guesser.model.load_state_dict(torch.load(
            os.path.join(directory, 'rnn.pt'), map_location=lambda storage, loc: storage
        ))
        guesser.model.eval()
        if CUDA:
            guesser.model = guesser.model.cuda()
        return guesser

    @classmethod
    def targets(cls):
        return ['rnn.pt', 'rnn.pkl']
