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
from qanta import logging

from progress import ProgressBar
from trainer import Trainer
from iterator import QuestionIterator
from util import *
from models import *

log = logging.get(__name__)

class config:
    def __init__(self):
        self.n_hidden      = 200
        self.optimizer     = 'Adam'
        self.lr            = 1e-3
        self.max_grad_norm = 5
        self.batch_size    = 128
        self.log_dir       = 'mlp_buzzer.log'
        self.model_dir     = 'output/buzzer/mlp_buzzer.npz'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, default=-1)
    parser.add_argument('-l', '--load', action='store_true', default=False)
    parser.add_argument('-e', '--epochs', type=int, default=3)
    return parser.parse_args()

def main():
    cfg = config()
    args = parse_args()

    id2option, all_guesses = load_quizbowl()
    train_iter = QuestionIterator(all_guesses['dev'], id2option, batch_size=cfg.batch_size,
            only_hopeful=True)
    eval_iter = QuestionIterator(all_guesses['test'], id2option, batch_size=cfg.batch_size,
            only_hopeful=False)

    model = MLP(n_input=eval_iter.n_input, n_hidden=200, n_output=2, n_layers=3,
            dropout=0.2)

    if args.gpu != -1 and cuda.available:
        log.info('Using gpu', args.gpu)
        cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)

    if os.path.exists(cfg.model_dir) and args.load:
        log.info('Loading model {0}'.format(cfg.model_dir))
        chainer.serializers.load_npz(cfg.model_dir, model)

    trainer = Trainer(model, cfg.model_dir)
    trainer = trainer.run(train_iter, eval_iter, args.epochs)

if __name__ == '__main__':
    main()
