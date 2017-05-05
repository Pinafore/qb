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

from qanta.buzzer.progress import ProgressBar

def metric(prediction, ground_truth, mask):
    if prediction.shape != ground_truth.shape:
        raise ValueError("Shape of prediction does not match with ground truth.")
    if prediction.shape != mask.shape:
        raise ValueError("Shape of prediction does not match with mask.")
    stats = dict()
    match = ((prediction == ground_truth) * mask)
    positive = (ground_truth * mask)
    positive_match = (match * positive).sum()
    total = mask.sum()
    stats['acc'] = (match.sum() / total).tolist()
    stats['pos_acc'] = (positive_match / total).tolist()
    return stats

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

    def test(self, test_iter):
        buzzes = dict()
        progress_bar = ProgressBar(test_iter.size, unit_iteration=True)
        for i in range(test_iter.size):
            batch = test_iter.next_batch(self.model.xp)
            length, batch_size, _ = batch.vecs.shape
            ys = self.model(batch.vecs, train=False)
            ys = F.reshape(ys, (length, batch_size, -1))
            actions = F.argmax(ys, axis=2).data # length, batch
            actions = actions.T.tolist()
            for q, a in zip(batch.qids, actions):
                q = q.tolist()
                buzzes[q] = -1 if not any(a) else a.index(1)
            progress_bar(*test_iter.epoch_detail)
        test_iter.finalize(reset=True)
        progress_bar.finalize()
        return buzzes

    def evaluate(self, eval_iter):
        stats = defaultdict(lambda: 0)
        progress_bar = ProgressBar(eval_iter.size, unit_iteration=True)
        for i in range(eval_iter.size):
            batch = eval_iter.next_batch(self.model.xp)
            length, batch_size, _ = batch.vecs.shape
            ys = self.model(batch.vecs, train=False)
            ts = F.reshape(batch.results, (length * batch_size, ))
            mask = F.reshape(batch.mask, (length * batch_size, ))

            stats['loss'] += F.sum(F.select_item(ys, ts) * mask).data.tolist()
            batch_stats = metric(F.argmax(ys, axis=1).data, ts.data, mask.data)
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
            ts = F.reshape(batch.results, (length * batch_size, ))
            mask = F.reshape(batch.mask, (length * batch_size, ))
            loss = -F.sum(F.select_item(ys, ts) * mask.data)
            self.backprop(loss)

            stats['loss'] += -loss.data.tolist()
            batch_stats = metric(F.argmax(ys, axis=1).data, ts.data, mask.data)
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
            print('\n{0}'.format(epoch))
            if train_iter is not None:
                train_stats = self.train_one_epoch(train_iter, progress_bar)
                output = 'train '
                for k, v in train_stats.items():
                    output += '{0}: {1:.2f}  '.format(k, v)
                print(output)
            if eval_iter is not None:
                output = 'eval '
                eval_stats = self.evaluate(eval_iter)
                for k, v in eval_stats.items():
                    output += '{0}: {1:.2f}  '.format(k, v)
                print(output)
            if self.model_dir is not None:
                chainer.serializers.save_npz(self.model_dir, self.model)
