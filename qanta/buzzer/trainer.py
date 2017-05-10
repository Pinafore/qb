import os
import pickle
import numpy as np
import argparse
from collections import defaultdict, namedtuple

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable
from chainer import cuda

from qanta.guesser.abstract import AbstractGuesser
from qanta.util import constants as c
from qanta.buzzer.util import GUESSERS
from qanta import logging

from qanta.buzzer.progress import ProgressBar

N_GUESSERS = len(GUESSERS)
log = logging.get(__name__)

class Trainer(object):

    def __init__(self, model, model_dir=None):
        self.model = model
        self.model_dir = model_dir
        self.optimizer = chainer.optimizers.Adam()
        self.optimizer.setup(self.model)
        self.optimizer.add_hook(chainer.optimizer.GradientClipping(5))

    def backprop(self, loss):
        self.optimizer.target.cleargrads()
        self.optimizer.update(lossfun=lambda: loss)

    def loss(self, ys, ts, mask):
        # ys: [length * batch_size, n_guessers]
        # ts: [length * batch_size, n_guessers]
        # mask: [length * batch_size]
        xp = self.model.xp
        ts = xp.asarray(ts.data, dtype=xp.float32)
        ys = F.log_softmax(ys) # length * batch_size, n_guessers
        loss = -F.sum(F.sum(ys * ts, axis=1) * mask.data) / mask.data.sum()
        return loss

    def metric(self, ys, ts, mask):
        # shapes are length * batch_size * n_guessers
        if ys.shape != ts.shape:
            raise ValueError("Shape of prediction {0} does not match with ground \
                truth {1}.".format( ys.shape, ts.shape))
        if ys.shape[0] != mask.shape[0]:
            raise ValueError("Shape0 of prediction {0} does not match with \
                mask0 {1}.".format(ys.shape[0], mask.shape[0]))
        stats = dict()
        ys = F.argmax(ys, axis=1)
        ts = self.model.xp.asarray(ts, dtype=self.model.xp.float32)
        correct = F.sum((F.select_item(ts, ys) * mask)).data
        total = mask.sum()
        stats['acc'] = (correct / total).tolist()
        return stats

    def test(self, test_iter):
        buzzes = dict()
        finals = dict()
        progress_bar = ProgressBar(test_iter.size, unit_iteration=True)
        for i in range(test_iter.size):
            batch = test_iter.next_batch(self.model.xp)
            length, batch_size, _ = batch.vecs.shape
            ys = self.model(batch.vecs, train=False)
            ys = F.swapaxes(F.reshape(ys, (length, batch_size, -1)), 0, 1)
            ys.to_cpu()
            ys = ys.data
            actions = np.argmax(ys, axis=2).tolist() # length, batch
            for qnum, action in zip(batch.qids, actions):
                if isinstance(qnum, np.ndarray):
                    qnum = qnum.tolist()
                buzzes[qnum] = [-1, -1]
                for i, a in enumerate(action):
                    if a < N_GUESSERS:
                        buzzes[qnum] = (i, a)
                        break
                finals[qnum] = np.argmax(ys[-1][:N_GUESSERS]).tolist()
            progress_bar(*test_iter.epoch_detail)
        test_iter.finalize(reset=True)
        progress_bar.finalize()
        return buzzes, finals

    def evaluate(self, eval_iter):
        stats = defaultdict(lambda: 0)
        progress_bar = ProgressBar(eval_iter.size, unit_iteration=True)
        for i in range(eval_iter.size):
            batch = eval_iter.next_batch(self.model.xp)
            length, batch_size, _ = batch.vecs.shape
            ys = self.model(batch.vecs, train=False)
            ts = F.reshape(batch.results, (length * batch_size, -1))
            mask = F.reshape(batch.mask, (length * batch_size, ))
            stats['loss'] = self.loss(ys, ts, mask).data.tolist()
            batch_stats = self.metric(ys.data, ts.data, mask.data)
            for k, v in batch_stats.items():
                stats[k] += v

            progress_bar(*eval_iter.epoch_detail)
        eval_iter.finalize(reset=True)
        progress_bar.finalize()

        for k, v in stats.items():
            stats[k] = v / eval_iter.size
        return stats

    def train_one_epoch(self, train_iter, progress_bar=None):
        stats = defaultdict(lambda: 0)
        for i in range(train_iter.size):
            batch = train_iter.next_batch(self.model.xp)
            length, batch_size, _ = batch.vecs.shape
            ys = self.model(batch.vecs, train=True)
            ts = F.reshape(batch.results, (length * batch_size, -1))
            mask = F.reshape(batch.mask, (length * batch_size, ))
            loss = self.loss(ys, ts, mask)
            self.backprop(loss)
            stats['loss'] = loss.data.tolist()
            batch_stats = self.metric(ys.data, ts.data, mask.data)
            for k, v in batch_stats.items():
                stats[k] += v

            if progress_bar is not None:
                progress_bar(*train_iter.epoch_detail)
        train_iter.finalize()
        if progress_bar is not None:
            progress_bar.finalize()

        for k, v in stats.items():
            stats[k] = v / train_iter.size
        return stats

    def run(self, train_iter=None, eval_iter=None, n_epochs=1):
        progress_bar = ProgressBar(n_epochs, unit_iteration=False)
        for epoch in range(n_epochs):
            log.info('\nepoch {0}'.format(epoch))
            if train_iter is not None:
                train_stats = self.train_one_epoch(train_iter, progress_bar)
                output = 'train '
                for k, v in train_stats.items():
                    output += '{0}: {1:.2f}  '.format(k, v)
                log.info(output)
            if eval_iter is not None:
                output = 'eval '
                eval_stats = self.evaluate(eval_iter)
                for k, v in eval_stats.items():
                    output += '{0}: {1:.2f}  '.format(k, v)
                log.info(output)
            if self.model_dir is not None:
                chainer.serializers.save_npz(self.model_dir, self.model)
