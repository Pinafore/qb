import os
import argparse
import chainer

from qanta import logging

from qanta.buzzer.progress import ProgressBar
from qanta.buzzer.trainer import Trainer
from qanta.buzzer.iterator import QuestionIterator
from qanta.buzzer.util import load_quizbowl
from qanta.buzzer.models import RNN, MLP

log = logging.get(__name__)

class config:
    def __init__(self):
        self.n_hidden      = 200
        self.optimizer     = 'Adam'
        self.lr            = 1e-3
        self.max_grad_norm = 5
        self.batch_size    = 128
        self.log_dir       = 'rnn_buzzer.log'
        self.model_dir     = 'output/buzzer/rnn_buzzer.npz'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, default=-1)
    parser.add_argument('-l', '--load', action='store_true', default=False)
    parser.add_argument('-s', '--save', action='store_true', default=False)
    parser.add_argument('-e', '--epochs', type=int, default=5)
    return parser.parse_args()

def main():
    cfg = config()
    args = parse_args()

    id2option, all_guesses = load_quizbowl()
    train_iter = QuestionIterator(all_guesses['dev'], id2option, batch_size=cfg.batch_size,
            only_hopeful=False)
    eval_iter = QuestionIterator(all_guesses['test'], id2option, batch_size=cfg.batch_size,
            only_hopeful=False)

    model = RNN(eval_iter.n_input, 128, 2)

    if args.gpu != -1 and chainer.cuda.available:
        log.info('Using gpu {0}'.format(args.gpu))
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)

    if os.path.exists(cfg.model_dir) and args.load:
        log.info('Loading model {0}'.format(cfg.model_dir))
        chainer.serializers.load_npz(cfg.model_dir, model)

    trainer = Trainer(model, cfg.model_dir)
    trainer = trainer.run(train_iter, eval_iter, args.epochs)

if __name__ == '__main__':
    main()
