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
from qanta.buzzer.util import load_quizbowl, GUESSERS
from qanta.buzzer.trainer import Trainer
from qanta.buzzer.models import MLP, RNN
from qanta.buzzer import constants as bc
from qanta.util import constants as c


log = logging.get(__name__)

def generate(fold, config):
    N_GUESSERS = len(GUESSERS)

    cfg = getattr(configs, config)()

    option2id, all_guesses = load_quizbowl([fold])
    test_iter = QuestionIterator(all_guesses[fold], option2id,
            batch_size=cfg.batch_size)
    
    if not os.path.exists(cfg.model_dir):
        log.info('Model {0} not available'.format(cfg.model_dir))
        exit(0)

    cfg = pickle.load(open(cfg.ckp_dir, 'rb'))

    if isinstance(cfg, configs.mlp):
        model = MLP(n_input=test_iter.n_input, n_hidden=cfg.n_hidden,
                n_output=N_GUESSERS+1, n_layers=cfg.n_layers,
                dropout=cfg.dropout, batch_norm=cfg.batch_norm)

    if isinstance(cfg, configs.rnn):
        model = RNN(test_iter.n_input, cfg.n_hidden, 2)

    log.info('Loading model {0}'.format(cfg.model_dir))
    chainer.serializers.load_npz(cfg.model_dir, model)

    gpu = conf['buzzer']['gpu']
    if gpu != -1 and chainer.cuda.available:
        log.info('Using gpu {0}'.format(gpu))
        chainer.cuda.get_device(gpu).use()
        model.to_gpu(gpu)

    trainer = Trainer(model, cfg.model_dir)
    buzzes = trainer.test(test_iter)
    log.info('Buzzes generated. Size {0}.'.format(len(buzzes)))

    buzzes_dir = bc.BUZZES_DIR.format(fold)
    with open(buzzes_dir, 'wb') as outfile:
        pickle.dump(buzzes, outfile)
    log.info('Buzzes saved to {0}.'.format(buzzes_dir))

    guesses_df = AbstractGuesser.load_guesses(bc.GUESSES_DIR, folds=[fold])
    buzzer2vwexpo(guesses_df, buzzes, fold)
    # preds, metas = buzzer2predsmetas(guesses_df, buzzes)
    # log.info('preds and metas generated')
    # performance.generate(2, preds, metas, 'output/summary/{}_1.json'.format(fold))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fold', default=None)
    parser.add_argument('-c', '--config', type=str, default='mlp')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args.fold is not None:
        folds = [args.fold]
    else:
        folds = c.BUZZER_GENERATION_FOLDS
    for fold in folds:
        log.info("Generating {} outputs".format(fold))
        generate(fold, args.config)
