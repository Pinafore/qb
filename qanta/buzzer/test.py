import os
import sys
import pickle
import numpy as np
import argparse
from collections import  namedtuple

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable
from chainer import cuda

from qanta.guesser.abstract import AbstractGuesser
from qanta.util import constants as c
from qanta import logging

from qanta.buzzer.interface buzzer2vwexpo
from qanta.buzzer.progress import ProgressBar
from qanta.buzzer.iterator import QuestionIterator
from qanta.buzzer.util import load_quizbowl
from qanta.buzzer.trainer import Trainer
from qanta.buzzer.models import MLP, RNN

log = logging.get(__name__)

class config:
    def __init__(self):
        self.n_hidden      = 200
        self.optimizer     = 'Adam'
        self.lr            = 1e-3
        self.max_grad_norm = 5
        self.batch_size    = 128
        self.guesses_dir   = 'data/guesses/'
        self.options_dir   = 'data/options.pickle'
        self.model_dir     = 'output/buzzer/mlp_buzzer.npz'
        self.log_dir       = 'mlp_buzzer.log'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fold', required=True)
    parser.add_argument('-g', '--gpu', type=int, default=-1)
    return parser.parse_args()

def main():
    cfg = config()
    args = parse_args()
    fold = args.fold

    id2option, all_guesses = load_quizbowl(cfg)
    test_iter = QuestionIterator(all_guesses[fold], id2option, batch_size=cfg.batch_size)
    
    if not os.path.exists(cfg.model_dir):
        log.info('Model {0} not available'.format(cfg.model_dir))
        exit(0)

    model = MLP(n_input=test_iter.n_input, n_hidden=200, n_output=2, n_layers=3,
            dropout=0.2)

    log.info('Loading model {0}'.format(cfg.model_dir))
    chainer.serializers.load_npz(cfg.model_dir, model)

    if cuda.available and args.gpu != -1:
        log.info('Using gpu', args.gpu)
        cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)

    trainer = Trainer(model, cfg.model_dir)
    buzzes = trainer.test(test_iter)
    log.info('Buzzes generated')
    guesses_df = AbstractGuesser.load_guesses(cfg.guesses_dir, folds=[fold])
    # preds, metas = buzzer2predsmetas(guesses_df, buzzes)
    # log.info('preds and metas generated')
    # performance.generate(2, preds, metas, 'output/summary/{}_1.json'.format(fold))
    buzzer2vwexpo(guesses_df, buzzes, fold)

if __name__ == '__main__':
    main()
