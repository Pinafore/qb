import os
import argparse
import chainer
import pickle
from collections import defaultdict, namedtuple
from typing import List, Dict, Tuple, Optional

from qanta import qlogging
from qanta.config import conf
from qanta.guesser.abstract import AbstractGuesser

from qanta.buzzer import configs
from qanta.buzzer.progress import ProgressBar
from qanta.buzzer.trainer import Trainer
from qanta.buzzer.iterator import QuestionIterator
from qanta.buzzer import iterator 
from qanta.buzzer.util import load_quizbowl, GUESSERS
from qanta.buzzer.models import MLP, RNN
from qanta.buzzer import constants as bc
from qanta.util import constants as c

log = qlogging.get(__name__)

def train_cost_sensitive(config, folds):
    N_GUESSERS = len(GUESSERS)
    cfg = getattr(configs, config)()
    make_vector = getattr(iterator, cfg.make_vector)

    option2id, all_guesses = load_quizbowl()
    iterators = dict()
    for fold in c.BUZZER_INPUT_FOLDS:
        iterators[fold] = QuestionIterator(all_guesses[fold], option2id,
            batch_size=cfg.batch_size, make_vector=make_vector)

    model = RNN(eval_iter.n_input, cfg.n_hidden, N_GUESSERS + 1)

    gpu = conf['buzzer']['gpu']
    if gpu != -1 and chainer.cuda.available:
        log.info('Using gpu {0}'.format(gpu))
        chainer.cuda.get_device(gpu).use()
        model.to_gpu(gpu)

    pickle.dump(cfg, open(cfg.ckp_dir, 'wb'))
    trainer = Trainer(model, cfg.model_dir)
    trainer.run(iterators[c.BUZZER_TRAIN_FOLD], iterators[c.BUZZER_DEV_FOLD], 25)

    for fold in folds:
        test_iter = iterators[fold]
        buzzes = trainer.test(test_iter)
        log.info('{0} buzzes generated. Size {1}.'.format(fold, len(buzzes)))
        buzzes_dir = bc.BUZZES_DIR.format(fold, cfg.model_name)
        with open(buzzes_dir, 'wb') as outfile:
            pickle.dump(buzzes, outfile)
        log.info('Buzzes saved to {0}.'.format(buzzes_dir))

        if fold == 'expo':
            buzzer2vwexpo(all_guesses['expo'], buzzes, fold)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='rnn_200_1')
    parser.add_argument('-f', '--fold', type=str, default=None)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args.fold is None:
        folds = c.BUZZER_GENERATION_FOLDS
    else:
        folds = [args.fold]
    train_cost_sensitive(args.config, folds)
