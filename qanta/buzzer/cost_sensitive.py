import os
import argparse
import chainer

from qanta import logging
from qanta.config import conf

from qanta.buzzer import configs
from qanta.buzzer.progress import ProgressBar
from qanta.buzzer.trainer import Trainer
from qanta.buzzer.iterator import QuestionIterator
from qanta.buzzer.util import load_quizbowl
from qanta.buzzer.models import MLP, RNN

log = logging.get(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='mlp')
    parser.add_argument('-l', '--load', action='store_true', default=False)
    parser.add_argument('-e', '--epochs', type=int, default=3)
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = getattr(configs, args.config)()

    option2id, all_guesses = load_quizbowl()
    train_iter = QuestionIterator(all_guesses['dev'], 
            option2id, batch_size=cfg.batch_size, only_hopeful=True)
    eval_iter = QuestionIterator(all_guesses['test'], 
            option2id, batch_size=cfg.batch_size, only_hopeful=False)

    if isinstance(cfg, configs.mlp):
        model = MLP(n_input=eval_iter.n_input, n_hidden=cfg.n_hidden,
                n_output=2, n_layers=cfg.n_layers, dropout=cfg.dropout)

    if isinstance(cfg, configs.rnn):
        model = RNN(eval_iter.n_input, cfg.n_hidden, 2)

    gpu = conf['buzzer']['gpu']
    if gpu != -1 and chainer.cuda.available:
        log.info('Using gpu {0}'.format(gpu))
        chainer.cuda.get_device(gpu).use()
        model.to_gpu(gpu)

    if os.path.exists(cfg.model_dir) and args.load:
        log.info('Loading model {0}'.format(cfg.model_dir))
        chainer.serializers.load_npz(cfg.model_dir, model)

    trainer = Trainer(model, cfg.model_dir)
    trainer = trainer.run(train_iter, eval_iter, args.epochs)

if __name__ == '__main__':
    main()
