import os
import argparse
import pickle
import chainer

from qanta.guesser.abstract import AbstractGuesser
from qanta import logging
from qanta.config import conf

from qanta.buzzer import configs
from qanta.buzzer.interface import buzzer2vwexpo
from qanta.buzzer.iterator import QuestionIterator
from qanta.buzzer.util import load_quizbowl, GUESSERS, stupid_buzzer
from qanta.buzzer.trainer import Trainer
from qanta.buzzer.models import MLP, RNN
from qanta.buzzer import constants as bc
from qanta.util import constants as c


log = logging.get(__name__)

def generate(config, folds):
    N_GUESSERS = len(GUESSERS)
    option2id, all_guesses = load_quizbowl(folds)

    cfg = getattr(configs, config)()
    cfg = pickle.load(open(cfg.ckp_dir, 'rb'))

    iterators = dict()
    for fold in folds:
        iterators[fold] = QuestionIterator(all_guesses[fold], option2id,
            batch_size=cfg.batch_size)
    
    if not os.path.exists(cfg.model_dir):
        log.info('Model {0} not available'.format(cfg.model_dir))
        exit(0)

    if isinstance(cfg, configs.mlp):
        model = MLP(n_input=iterators[folds[0]].n_input, n_hidden=cfg.n_hidden,
                n_output=N_GUESSERS+1, n_layers=cfg.n_layers,
                dropout=cfg.dropout, batch_norm=cfg.batch_norm)

    if isinstance(cfg, configs.rnn):
        model = RNN(iterators[folds[0]].n_input, cfg.n_hidden, N_GUESSERS + 1)

    log.info('Loading model {0}'.format(cfg.model_dir))
    chainer.serializers.load_npz(cfg.model_dir, model)

    gpu = conf['buzzer']['gpu']
    if gpu != -1 and chainer.cuda.available:
        log.info('Using gpu {0}'.format(gpu))
        chainer.cuda.get_device(gpu).use()
        model.to_gpu(gpu)

    trainer = Trainer(model, cfg.model_dir)

    for fold in folds:
        buzzes = trainer.test(iterators[fold])
        log.info('{0} buzzes generated. Size {1}.'.format(fold, len(buzzes)))
        buzzes_dir = bc.BUZZES_DIR.format(fold, cfg.model_name)
        with open(buzzes_dir, 'wb') as f:
            pickle.dump(buzzes, f)
        log.info('Buzzes saved to {0}.'.format(buzzes_dir))

        if fold == 'expo':
            guesses_df = AbstractGuesser.load_guesses(bc.GUESSES_DIR, folds=[fold])
            buzzer2vwexpo(guesses_df, buzzes, fold)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fold', default=None)
    parser.add_argument('-c', '--config', type=str, default='rnn')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args.fold is not None:
        folds = [args.fold]
    else:
        folds = c.BUZZER_GENERATION_FOLDS
        log.info("Generating {} outputs".format(folds))
    generate(args.config, folds)
