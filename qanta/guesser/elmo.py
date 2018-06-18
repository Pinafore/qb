from typing import List, Optional, Tuple
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.optim import Adam, lr_scheduler
from allennlp.modules.elmo import Elmo, batch_to_ids

from qanta.datasets.abstract import QuestionText, Page, TrainingData
from qanta.guesser.abstract import AbstractGuesser
from qanta.preprocess import preprocess_dataset
from qanta.util.io import get_tmp_filename
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
    def __init__(self, n_classes, elmo_dropout=.5, sm_dropout=.15):
        super().__init__()
        self.elmo_dropout = elmo_dropout
        self.sm_dropout = sm_dropout
        self.elmo = Elmo(ELMO_OPTIONS_FILE, ELMO_WEIGHTS_FILE, 2, dropout=elmo_dropout, requires_grad=False)
        self.classifier = nn.Sequential(
            nn.Linear(2 * ELMO_DIM, n_classes),
            nn.BatchNorm1d(n_classes),
            nn.Dropout(sm_dropout)
        )

    def forward(self, questions):
        embeddings = self.elmo(questions)
        layer_0 = embeddings['elmo_representations'][0]
        layer_0 = layer_0.sum(1) / layer_0.shape[1]
        layer_1 = embeddings['elmo_representations'][1]
        layer_1 = layer_1.sum(1) / layer_1.shape[1]
        layer = torch.cat([layer_0, layer_1], 1)
        return self.classifier(layer)


def batchify(x_data, y_data, batch_size=128, shuffle=False):
    batches = []
    for i in range(0, len(x_data), batch_size):
        start, stop = i, i + batch_size
        x_batch = batch_to_ids(x_data[start:stop])
        if CUDA:
            y_batch = Variable(torch.from_numpy(np.array(y_data[start:stop])).cuda())
        else:
            y_batch = Variable(torch.from_numpy(np.array(y_data[start:stop])))
        batches.append((x_batch, y_batch))

    if shuffle:
        random.shuffle(batches)

    return batches


class ElmoGuesser(AbstractGuesser):
    def __init__(self, config_num):
        super(ElmoGuesser, self).__init__(config_num)
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
        self.model = ElmoModel(len(i_to_class))
        if CUDA:
            self.model = self.model.cuda()
        log.info(f'Parameters:\n{self.parameters()}')
        log.info(f'Model:\n{self.model}')
        self.optimizer = Adam([p for p in self.model.parameters() if p.requires_grad])
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
        for x_batch, y_batch in batches:
            if train:
                self.model.zero_grad()
            out = self.model(x_batch.cuda())
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
        pass

    @classmethod
    def targets(cls) -> List[str]:
        return ['elmo.pt', 'elmo.pkl']

    @classmethod
    def load(cls, directory: str):
        pass

    def save(self, directory: str) -> None:
        pass

