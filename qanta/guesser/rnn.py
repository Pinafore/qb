import re
import os
import shutil
import time
import cloudpickle
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
from qanta.util.io import shell, get_tmp_filename
from qanta.torch.dataset import QuizBowl
from qanta.config import conf
from qanta.guesser.abstract import AbstractGuesser
from qanta.datasets.abstract import QuestionText
from qanta.torch import (
    BaseLogger, TerminateOnNaN, EarlyStopping, ModelCheckpoint,
    MaxEpochStopping, TrainingManager
)


log = qlogging.get(__name__)


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


class RnnModel(nn.Module):
    def __init__(self, n_classes, *,
                 text_field=None,
                 init_embeddings=True, emb_dim=300,
                 n_hidden_units=1000, n_hidden_layers=1,
                 nn_dropout=.265, sm_dropout=.158, bidirectional=True):
        super(RnnModel, self).__init__()
        self.emb_dim = emb_dim
        self.n_classes = n_classes
        self.n_hidden_units = n_hidden_units
        self.n_hidden_layers = n_hidden_layers
        self.nn_dropout = nn_dropout
        self.sm_dropout = sm_dropout
        self.bidirectional = bidirectional
        self.num_directions = 1 + int(bidirectional)

        self.dropout = nn.Dropout(nn_dropout)

        text_vocab = text_field.vocab
        self.text_vocab_size = len(text_vocab)
        text_pad_idx = text_vocab.stoi[text_field.pad_token]
        self.text_embeddings = nn.Embedding(self.text_vocab_size, emb_dim, padding_idx=text_pad_idx)
        self.text_field = text_field
        if init_embeddings:
            mean_emb = text_vocab.vectors.mean(0)
            text_vocab.vectors[text_vocab.stoi[text_field.unk_token]] = mean_emb
            self.text_embeddings.weight.data = text_vocab.vectors.cuda()

        self.rnn = nn.GRU(
            self.emb_dim, n_hidden_units, n_hidden_layers,
            dropout=self.nn_dropout, batch_first=True, bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.n_hidden_units, n_classes),
            nn.BatchNorm1d(n_classes),
            nn.Dropout(self.sm_dropout)
        )

    def forward(self,
                text_input: Variable,
                lengths: List[int],
                hidden: Variable,
                qanta_ids):
        """
        :param text_input: [batch_size, seq_len] of word indices
        :param lengths: Length of each example
        :param qanta_ids: QB qanta_id if a qb question, otherwise -1 for wikipedia, used to get domain as source/target
        :param hidden: hidden state
        :return:
        """
        embed = self.text_embeddings(text_input)
        embed = self.dropout(embed)

        packed_input = nn.utils.rnn.pack_padded_sequence(embed, lengths, batch_first=True)
        output, hidden = self.rnn(packed_input, hidden)

        if type(hidden) == tuple:
            final_hidden = hidden[0]
        else:
            final_hidden = hidden

        batch_size = text_input.data.shape[0]

        # Since number of layers is variable, we need a way to reduce this
        # to just one output. The easiest is to take the last hidden, but
        # we could try other things too.
        final_hidden = final_hidden.view(
            self.n_hidden_layers, self.num_directions, batch_size, self.n_hidden_units
        )[-1].view(self.num_directions, batch_size, self.n_hidden_units)

        return self.classifier(final_hidden), hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        if isinstance(self.rnn, nn.LSTM):
            return (
                Variable(weight.new(
                    self.n_hidden_layers * self.num_directions, batch_size, self.n_hidden_units).zero_()),
                Variable(
                    weight.new(self.n_hidden_layers * self.num_directions, batch_size, self.n_hidden_units).zero_())
            )
        else:
            return Variable(weight.new(
                self.n_hidden_layers * self.num_directions, batch_size, self.n_hidden_units).zero_())


class RnnGuesser(AbstractGuesser):
    def __init__(self, config_num):
        super(RnnGuesser, self).__init__(config_num)
        if self.config_num is not None:
            guesser_conf = conf['guessers']['qanta.guesser.rnn.RnnGuesser'][self.config_num]
            self.gradient_clip = guesser_conf['gradient_clip']
            self.n_hidden_units = guesser_conf['n_hidden_units']
            self.n_hidden_layers = guesser_conf['n_hidden_layers']
            self.lr = guesser_conf['lr']
            self.nn_dropout = guesser_conf['nn_dropout']
            self.sm_dropout = guesser_conf['sm_dropout']
            self.batch_size = guesser_conf['batch_size']
            self.use_wiki = guesser_conf['use_wiki']
            self.n_wiki_sentences = guesser_conf['n_wiki_sentences']
            self.wiki_title_replace_token = guesser_conf['wiki_title_replace_token']
            self.lowercase = guesser_conf['lowercase']

            self.random_seed = guesser_conf['random_seed']

        self.page_field: Optional[Field] = None
        self.qanta_id_field: Optional[Field] = None
        self.text_field: Optional[Field] = None
        self.n_classes = None
        self.emb_dim = None
        self.model_file = None

        self.model: Optional[RnnModel] = None
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
        return conf['guessers']['qanta.guesser.rnn.RnnGuesser'][self.config_num]

    def train(self, training_data):
        log.info('Loading Quiz Bowl dataset')
        train_iter, val_iter, dev_iter = QuizBowl.iters(
            batch_size=self.batch_size, lower=self.lowercase,
            use_wiki=self.use_wiki, n_wiki_sentences=self.n_wiki_sentences,
            replace_title_mentions=self.wiki_title_replace_token
        )
        log.info(f'N Train={len(train_iter.dataset.examples)}')
        log.info(f'N Test={len(val_iter.dataset.examples)}')
        fields: Dict[str, Field] = train_iter.dataset.fields
        self.page_field = fields['page']
        self.n_classes = len(self.ans_to_i)
        self.qanta_id_field = fields['qanta_id']
        self.emb_dim = 300

        self.text_field = fields['text']
        log.info(f'Text Vocab={len(self.text_field.vocab)}')

        log.info('Initializing Model')
        self.model = RnnModel(
            self.n_classes,
            text_field=self.text_field,
            emb_dim=self.emb_dim,
            n_hidden_units=self.n_hidden_units, n_hidden_layers=self.n_hidden_layers,
            nn_dropout=self.nn_dropout, sm_dropout=self.sm_dropout
        )
        if CUDA:
            self.model = self.model.cuda()
        log.info(f'Parameters:\n{self.parameters()}')
        log.info(f'Model:\n{self.model}')
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, verbose=True, mode='max')

        temp_prefix = get_tmp_filename()
        self.model_file = f'{temp_prefix}.pt'
        manager = TrainingManager([
            BaseLogger(log_func=log.info), TerminateOnNaN(), EarlyStopping(monitor='test_acc', patience=10, verbose=1),
            MaxEpochStopping(100), ModelCheckpoint(create_save_model(self.model), self.model_file, monitor='test_acc')
        ])

        log.info('Starting training')

        epoch = 0
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
            epoch += 1

    def run_epoch(self, iterator: Iterator):
        is_train = iterator.train
        batch_accuracies = []
        batch_losses = []
        hidden_init = self.model.init_hidden(self.batch_size)
        epoch_start = time.time()
        for batch in iterator:
            text, lengths = batch.text

            page = batch.page
            qanta_ids = batch.qanta_id.cuda()

            if is_train:
                self.model.zero_grad()

            out, hidden = self.model(
                text, lengths, hidden_init, qanta_ids
            )
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
        if len(questions) == 0:
            return []
        batch_size = 128
        if len(questions) < batch_size:
            return self._guess_batch(questions, max_n_guesses)
        else:
            all_guesses = []
            for i in range(0, len(questions), batch_size):
                batch_questions = questions[i:i + batch_size]
                guesses = self._guess_batch(batch_questions, max_n_guesses)
                all_guesses.extend(guesses)
            return all_guesses

    def _guess_batch(self, questions: List[QuestionText], max_n_guesses: Optional[int]):
        if len(questions) == 0:
            return []
        examples = [self.text_field.preprocess(q) for q in questions]
        text, lengths = self.text_field.process(examples, None, False)

        qanta_ids = self.qanta_id_field.process([0 for _ in questions]).cuda()
        guesses = []
        hidden_init = self.model.init_hidden(len(questions))
        out = self.model(text, lengths, hidden_init, qanta_ids)
        probs = F.softmax(out).data.cpu().numpy()
        n_examples = probs.shape[0]
        preds = np.argsort(-probs, axis=1)
        for i in range(n_examples):
            guesses.append([])
            for p in preds[i][:max_n_guesses]:
                guesses[-1].append((self.i_to_ans[p], probs[i][p]))
        return guesses

    def save(self, directory: str):
        shutil.copyfile(self.model_file, os.path.join(directory, 'rnn.pt'))
        shell(f'rm -f {self.model_file}')
        with open(os.path.join(directory, 'rnn.pkl'), 'wb') as f:
            cloudpickle.dump({
                'page_field': self.page_field,
                'text_field': self.text_field,
                'qanta_id_field': self.qanta_id_field,
                'n_classes': self.n_classes,
                'gradient_clip': self.gradient_clip,
                'n_hidden_units': self.n_hidden_units,
                'n_hidden_layers': self.n_hidden_layers,
                'lr': self.lr,
                'nn_dropout': self.nn_dropout,
                'sm_dropout': self.sm_dropout,
                'batch_size': self.batch_size,
                'use_wiki': self.use_wiki,
                'n_wiki_sentences': self.n_wiki_sentences,
                'wiki_title_replace_token': self.wiki_title_replace_token,
                'lowercase': self.lowercase,
                'random_seed': self.random_seed,
                'config_num': self.config_num
            }, f)

    @classmethod
    def load(cls, directory: str):
        with open(os.path.join(directory, 'rnn.pkl'), 'rb') as f:
            params = cloudpickle.load(f)

        guesser = RnnGuesser(params['config_num'])
        guesser.page_field = params['page_field']
        guesser.qanta_id_field = params['qanta_id_field']

        guesser.text_field = params['text_field']

        guesser.n_classes = params['n_classes']
        guesser.gradient_clip = params['gradient_clip']
        guesser.n_hidden_units = params['n_hidden_units']
        guesser.n_hidden_layers = params['n_hidden_layers']
        guesser.lr = params['lr']
        guesser.nn_dropout = params['nn_dropout']
        guesser.sm_dropout = params['sm_dropout']
        guesser.use_wiki = params['use_wiki']
        guesser.n_wiki_sentences = params['n_wiki_sentences']
        guesser.wiki_title_replace_token = params['wiki_title_replace_token']
        guesser.lowercase = params['lowercase']
        guesser.random_seed = params['random_seed']
        guesser.model = RnnModel(
            guesser.n_classes,
            text_field=guesser.text_field,
            init_embeddings=False, emb_dim=300,
            n_hidden_layers=guesser.n_hidden_layers,
            n_hidden_units=guesser.n_hidden_units
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
