import sys
import pickle
import chainer

from qanta.guesser.abstract import AbstractGuesser
from qanta.config import conf

from qanta.buzzer import configs
from qanta.util import constants as c
from qanta.datasets.quiz_bowl import QuestionDatabase
from qanta.buzzer.iterator import QuestionIterator
from qanta.buzzer.util import load_quizbowl, GUESSERS
from qanta.buzzer.trainer import Trainer
from qanta.buzzer.models import MLP, RNN
from qanta.buzzer import constants as bc
from qanta.reporting.new_performance import generate as generate_report
from qanta import logging

N_GUESSERS = len(GUESSERS)
log = logging.get(__name__)


def run(cfg, fold, all_guesses, option2id):
    train_iter = QuestionIterator(all_guesses['dev'], option2id,
            batch_size=cfg.batch_size, step_size=cfg.step_size,
            neg_weight=cfg.neg_weight)
    test_iter = QuestionIterator(all_guesses[fold], option2id,
            batch_size=cfg.batch_size, step_size=cfg.step_size,
            neg_weight=cfg.neg_weight)

    model = MLP(n_input=test_iter.n_input, n_hidden=cfg.n_hidden,
            n_output=N_GUESSERS+1, n_layers=cfg.n_layers,
            dropout=cfg.dropout, batch_norm=cfg.batch_norm)

    gpu = conf['buzzer']['gpu']
    if gpu != -1 and chainer.cuda.available:
        log.info('Using gpu {0}'.format(gpu))
        chainer.cuda.get_device(gpu).use()
        model.to_gpu(gpu)

    trainer = Trainer(model, cfg.model_dir)
    trainer.run(train_iter, test_iter, 4, verbose=True)

    buzzes = trainer.test(test_iter)
    log.info('Buzzes generated. Size {0}.'.format(len(buzzes)))

    return buzzes

def get_cfgs():
    # n_layers, n_hidden, batch_norm / dropout, neg_weight, # step_size
    _n_layers = [1,2,3]
    _n_hidden = [50, 100, 200]
    _batch_norm = [True, False]
    _neg_weight = [1, 0.95, 0.9, 0.85, 0.8]
    cfgs = []
    for n_layers in _n_layers:
        cfg = configs.mlp()
        cfg.n_layers = n_layers
        cfgs.append(cfg)
    for n_hidden in _n_hidden:
        cfg = configs.mlp()
        cfg.n_hidden = n_hidden
        cfgs.append(cfg)
    for batch_norm in _batch_norm:
        cfg = configs.mlp()
        cfg.batch_norm = batch_norm
        cfgs.append(cfg)
    for neg_weight in _neg_weight:
        cfg = configs.mlp()
        cfg.neg_weight = neg_weight
        cfgs.append(cfg)
    return cfgs

def hyper_search(fold):
    option2id, all_guesses = load_quizbowl()

    all_questions = QuestionDatabase().all_questions()
    answers = {k: v.page for k, v in all_questions.items()}
    guesses_df = AbstractGuesser.load_guesses(bc.GUESSES_DIR, folds=[fold])

    cfgs = get_cfgs()
    cfg_buzzes = []
    for i, cfg in enumerate(cfgs):
        print('**********{}**********'.format(i))
        buzzes = run(cfg, fold, all_guesses, option2id)
        cfg_buzzes.append((cfg, buzzes))

    with open('output/buzzer/cfg_buzzes_{}.pkl'.format(fold), 'wb') as outfile:
        pickle.dump(cfg_buzzes, outfile)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        folds = [sys.argv[1]]
    else:
        folds = c.BUZZER_GENERATION_FOLDS

    for fold in folds:
        hyper_search(fold)
