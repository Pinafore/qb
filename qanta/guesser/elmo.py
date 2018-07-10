from typing import List, Optional, Tuple
import os
import shutil
import random
import time

import numpy as np
import cloudpickle
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.optim import Adam, lr_scheduler
from allennlp.modules.elmo import Elmo, batch_to_ids

from qanta.datasets.abstract import QuestionText, Page, TrainingData
from qanta.guesser.abstract import AbstractGuesser
from qanta.preprocess import preprocess_dataset, tokenize_question
from qanta.util.io import get_tmp_filename, shell
from qanta.config import conf
from qanta.torch import (
    BaseLogger, TerminateOnNaN, EarlyStopping, ModelCheckpoint,
    MaxEpochStopping, TrainingManager
)
from qanta import qlogging


log = qlogging.get(__name__)

ELMO_OPTIONS_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
ELMO_WEIGHTS_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
ELMO_DIM = 1024
CUDA = torch.cuda.is_available()


def create_save_model(model):
    def save_model(path):
        torch.save(model.state_dict(), path)
    return save_model


class ElmoModel(nn.Module):
    def __init__(self, n_classes, dropout=.5, unfreeze=None):
        super().__init__()
        self.dropout = dropout
        self.unfreeze = unfreeze
        # This turns off gradient updates for the elmo model, but still leaves scalar mixture
        # parameters as tunable, provided that references to the scalar mixtures are extracted
        # and plugged into the optimizer
        self.elmo = Elmo(ELMO_OPTIONS_FILE, ELMO_WEIGHTS_FILE, 2, dropout=dropout, requires_grad=False)
        self.classifier = nn.Sequential(
            nn.Linear(2 * ELMO_DIM, n_classes),
            nn.BatchNorm1d(n_classes),
            nn.Dropout(dropout)
        )

    def forward(self, questions, lengths):
        embeddings = self.elmo(questions)
        layer_0 = embeddings['elmo_representations'][0]
        layer_0 = layer_0.sum(1) / lengths
        layer_1 = embeddings['elmo_representations'][1]
        layer_1 = layer_1.sum(1) / lengths
        layer = torch.cat([layer_0, layer_1], 1)
        return self.classifier(layer)


def batchify(x_data, y_data, batch_size=128, shuffle=False):
    batches = []
    for i in range(0, len(x_data), batch_size):
        start, stop = i, i + batch_size
        x_batch = batch_to_ids(x_data[start:stop])
        lengths = torch.from_numpy(np.array([1.0 * len(x) for x in x_data[start:stop]]))
        if CUDA:
            y_batch = Variable(torch.from_numpy(np.array(y_data[start:stop])).cuda())
        else:
            y_batch = Variable(torch.from_numpy(np.array(y_data[start:stop])))
        batches.append((x_batch, y_batch, lengths))

    if shuffle:
        random.shuffle(batches)

    return batches


class ElmoGuesser(AbstractGuesser):
    def __init__(self, config_num):
        super(ElmoGuesser, self).__init__(config_num)
        if config_num is not None:
            guesser_conf = conf['guessers']['qanta.guesser.elmo.ElmoGuesser'][self.config_num]
            self.random_seed = guesser_conf['random_seed']
            self.dropout = guesser_conf['dropout']
        else:
            self.random_seed = None
            self.dropout = None

        self.model = None
        self.i_to_class = None
        self.class_to_i = None

        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        self.model_file = None

    def train(self, training_data: TrainingData) -> None:
        x_train, y_train, x_val, y_val, vocab, class_to_i, i_to_class = preprocess_dataset(training_data)
        self.class_to_i = class_to_i
        self.i_to_class = i_to_class

        log.info('Batchifying data')
        train_batches = batchify(x_train, y_train, shuffle=True)
        val_batches = batchify(x_val, y_val, shuffle=False)
        self.model = ElmoModel(len(i_to_class), dropout=self.dropout)
        if CUDA:
            self.model = self.model.cuda()
        log.info(f'Parameters:\n{self.parameters()}')
        log.info(f'Model:\n{self.model}')
        if self.elmo_unfreeze != 'never':
            parameters = list(self.model.classifier.parameters())
            for mix in self.model.elmo._scalar_mixes:
                parameters.extend(list(mix.parameters()))
            self.optimizer = Adam(parameters)
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
            if epoch == 10 and self.elmo_unfreeze == 'epoch_10':
                elmo_param_group = [pg for pg in self.optimizer.param_groups if pg['name'] == 'elmo'][0]
                elmo_param_group['lr'] = 0.001
            self.model.train()
            train_acc, train_loss, train_time = self.run_epoch(train_batches)
            random.shuffle(train_batches)

            self.model.eval()
            test_acc, test_loss, test_time = self.run_epoch(val_batches)

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

    def run_epoch(self, batches, train=True):
        batch_accuracies = []
        batch_losses = []
        epoch_start = time.time()
        for x_batch, y_batch, length_batch in batches:
            if train:
                self.model.zero_grad()
            out = self.model(x_batch.cuda(), length_batch.cuda())
            _, preds = torch.max(out, 1)
            accuracy = torch.mean(torch.eq(preds, y_batch).float()).data[0]
            batch_loss = self.criterion(out, y_batch)
            if train:
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm(self.model.parameters(), .25)
                self.optimizer.step()
            batch_accuracies.append(accuracy)
            batch_losses.append(batch_loss.data[0])
        epoch_end = time.time()

        return np.mean(batch_accuracies), np.mean(batch_losses), epoch_end - epoch_start

    def guess(self, questions: List[QuestionText], max_n_guesses: Optional[int]) -> List[List[Tuple[Page, float]]]:
        y_data = np.zeros((len(questions)))
        x_data = [tokenize_question(q) for q in questions]
        batches = batchify(x_data, y_data, shuffle=False)
        guesses = []
        for x_batch, y_batch, length_batch in batches:
            out = self.model(x_batch.cuda(), length_batch)
            probs = F.softmax(out).data.cpu().numpy()
            preds = np.argsort(-probs, axis=1)
            n_examples = probs.shape[0]
            for i in range(n_examples):
                example_guesses = []
                for p in preds[i][:max_n_guesses]:
                    example_guesses.append((self.i_to_class[p], probs[i][p]))
                guesses.append(example_guesses)

        return guesses

    @classmethod
    def targets(cls) -> List[str]:
        return ['elmo.pt', 'elmo.pkl']

    @classmethod
    def load(cls, directory: str):
        with open(os.path.join(directory, 'elmo.pkl'), 'rb') as f:
            params = cloudpickle.load(f)

        guesser = ElmoGuesser(params['config_num'])
        guesser.class_to_i = params['class_to_i']
        guesser.i_to_class = params['i_to_class']
        guesser.random_seed = params['random_seed']
        guesser.dropout = params['dropout']
        guesser.model = ElmoModel(len(guesser.i_to_class))
        guesser.model.load_state_dict(torch.load(
            os.path.join(directory, 'elmo.pt'), map_location=lambda storage, loc: storage
        ))
        guesser.model.eval()
        if CUDA:
            guesser.model = guesser.model.cuda()
        return guesser

    def save(self, directory: str) -> None:
        shutil.copyfile(self.model_file, os.path.join(directory, 'elmo.pt'))
        shell(f'rm -f {self.model_file}')
        with open(os.path.join(directory, 'elmo.pkl'), 'wb') as f:
            cloudpickle.dump({
                'class_to_i': self.class_to_i,
                'i_to_class': self.i_to_class,
                'config_num': self.config_num,
                'random_seed': self.random_seed,
                'dropout': self.dropout
            }, f)

