import os
import argparse
import pickle
import chainer

from qanta.guesser.abstract import AbstractGuesser
from qanta import logging
from qanta.config import conf
from qanta.preprocess import format_guess

from qanta.buzzer import configs
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
    trainer.run(train_iter, test_iter, 4)

    buzzes = trainer.test(test_iter)
    log.info('Buzzes generated. Size {0}.'.format(len(buzzes)))
    return buzzes

def hyper_search(fold):
    cfg = getattr(configs, 'mlp')()

    option2id, all_guesses = load_quizbowl()

    all_questions = QuestionDatabase().all_questions()
    answers = {k: format_guess(v.page) for k, v in all_questions.items()}
    guesses_df = AbstractGuesser.load_guesses(bc.GUESSES_DIR, folds=[fold])

    outfile = open('n_layers.txt', 'w')
    for i in [1,2,3,4,5]:
        cfg.n_layers = i
        outfile.write("n_layers: {0}\n".format(i))
        buzzes = run(cfg, fold, all_guesses, option2id)
        output = generate_report(buzzes, answers, guesses_df, fold)
        outfile.write(output + '\n')
        outfile.flush()
    outfile.close()

if __name__ == '__main__':
    fold = 'test'
    hyper_search(fold)
