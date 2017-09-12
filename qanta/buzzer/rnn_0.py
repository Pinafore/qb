import os
import numpy as np
import argparse
import chainer
import pickle
from collections import defaultdict, namedtuple
from typing import List, Dict, Tuple, Optional

from qanta import logging
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
from qanta.buzzer.report import report
from qanta.util import constants as c

N_GUESSERS = len(GUESSERS)
N_GUESSES = 10

log = logging.get(__name__)

def dense_vector0(dicts: List[List[Dict[str, float]]],
        step_size=1) -> List[List[float]]:
    length = len(dicts)
    prev_vec = [0. for _ in range(N_GUESSERS * N_GUESSES)]
            
    vecs = []
    for i in range(length):
        if len(dicts[i]) != N_GUESSERS:
            raise ValueError("Inconsistent number of guessers ({0}, {1}).".format(
                N_GUESSERS, len(dicts)))
        vec = []
        diff_vec = []
        isnew_vec = []
        for j in range(N_GUESSERS):
            dic = sorted(dicts[i][j].items(), key=lambda x: x[1],
                    reverse=True)[:N_GUESSES]
            for guess, score in dic:
                vec.append(score)
                if i > 0 and guess in dicts[i-1][j]:
                    diff_vec.append(score - dicts[i-1][j][guess])
                    isnew_vec.append(0)
                else:
                    diff_vec.append(score) 
                    isnew_vec.append(1)
            if len(dic) < N_GUESSES:
                for k in range(max(N_GUESSES - len(dic), 0)):
                    vec.append(0)
                    diff_vec.append(0)
                    isnew_vec.append(0)
        features = [
                vec[0], vec[1], vec[2],
                isnew_vec[0], isnew_vec[1], isnew_vec[2],
                diff_vec[0], diff_vec[1], diff_vec[2],
                vec[0] - vec[1], vec[1] - vec[2], 
                vec[0] - prev_vec[0],
                sum(isnew_vec[:5]),
                np.average(vec), np.average(prev_vec),
                np.average(vec[:5]), np.average(prev_vec[:5]),
                np.var(vec), np.var(prev_vec),
                np.var(vec[:5]), np.var(prev_vec[:5])
                ]

        vecs.append(features)
        prev_vec = vec
    return vecs

def dense_vector1(dicts: List[List[Dict[str, float]]],
        step_size=1) -> List[List[float]]:
    length = len(dicts)
    prev_vec = [0. for _ in range(N_GUESSERS * N_GUESSES)]
            
    vecs = []
    for i in range(length):
        if len(dicts[i]) != N_GUESSERS:
            raise ValueError("Inconsistent number of guessers ({0}, {1}).".format(
                N_GUESSERS, len(dicts)))
        vec = []
        diff_vec = []
        isnew_vec = []
        for j in range(N_GUESSERS):
            dic = sorted(dicts[i][j].items(), key=lambda x: x[1],
                    reverse=True)[:N_GUESSES]
            for guess, score in dic:
                vec.append(score)
                if i > 0 and guess in dicts[i-1][j]:
                    diff_vec.append(score - dicts[i-1][j][guess])
                    isnew_vec.append(0)
                else:
                    diff_vec.append(score) 
                    isnew_vec.append(1)
            if len(dic) < N_GUESSES:
                for k in range(max(N_GUESSES - len(dic), 0)):
                    vec.append(0)
                    diff_vec.append(0)
                    isnew_vec.append(0)
        features = [
                vec[0], vec[1],
                isnew_vec[0], isnew_vec[1],
                diff_vec[0], diff_vec[1],
                vec[0] - vec[1],
                vec[0] - prev_vec[0],
                np.average(vec[:5]), np.average(prev_vec[:5]),
                np.var(vec[:5]), np.var(prev_vec[:5])
                ]

        vecs.append(features)
        prev_vec = vec
    return vecs

def dense_vector2(dicts: List[List[Dict[str, float]]],
        step_size=1) -> List[List[float]]:
    length = len(dicts)
    prev_vec = [0. for _ in range(N_GUESSERS * N_GUESSES)]
            
    vecs = []
    for i in range(length):
        if len(dicts[i]) != N_GUESSERS:
            raise ValueError("Inconsistent number of guessers ({0}, {1}).".format(
                N_GUESSERS, len(dicts)))
        vec = []
        diff_vec = []
        isnew_vec = []
        for j in range(N_GUESSERS):
            dic = sorted(dicts[i][j].items(), key=lambda x: x[1],
                    reverse=True)[:N_GUESSES]
            for guess, score in dic:
                vec.append(score)
                if i > 0 and guess in dicts[i-1][j]:
                    diff_vec.append(score - dicts[i-1][j][guess])
                    isnew_vec.append(0)
                else:
                    diff_vec.append(score) 
                    isnew_vec.append(1)
            if len(dic) < N_GUESSES:
                for k in range(max(N_GUESSES - len(dic), 0)):
                    vec.append(0)
                    diff_vec.append(0)
                    isnew_vec.append(0)
        features = [
                vec[0], vec[1],
                isnew_vec[0], isnew_vec[1],
                diff_vec[0], diff_vec[1],
                vec[0] - vec[1],
                vec[0] - prev_vec[0],
                np.average(vec[:3]), np.average(prev_vec[:3]),
                np.var(vec[:3]), np.var(prev_vec[:3])
                ]

        vecs.append(features)
        prev_vec = vec
    return vecs

def main():
    np.random.seed(0)
    try: 
        import cupy
        cupy.random.seed(0)
    except Exception:
        pass

    option2id, all_guesses = load_quizbowl()
    train_iter = QuestionIterator(all_guesses[c.BUZZER_TRAIN_FOLD], option2id,
            batch_size=128, make_vector=dense_vector)
    dev_iter = QuestionIterator(all_guesses[c.BUZZER_DEV_FOLD], option2id,
            batch_size=128, make_vector=dense_vector)
    devtest_iter = QuestionIterator(all_guesses['test'], option2id,
            batch_size=128, make_vector=dense_vector)
    expo_iter = QuestionIterator(all_guesses['expo'], option2id,
            batch_size=128, make_vector=dense_vector)

    n_hidden = 300
    model_name = 'neo_0'
    model_dir = 'output/buzzer/neo/{}.npz'.format(model_name)
    model = RNN(train_iter.n_input, n_hidden, N_GUESSERS + 1)

    chainer.cuda.get_device(0).use()
    model.to_gpu(0)

    trainer = Trainer(model, model_dir)
    trainer.run(train_iter, dev_iter, 25)

    dev_buzzes = trainer.test(dev_iter)
    dev_buzzes_dir = 'output/buzzer/neo/dev_buzzes.{}.pkl'.format(model_name)
    with open(dev_buzzes_dir, 'wb') as f:
        pickle.dump(dev_buzzes, f)
    print('Dev buzz {} saved to {}'.format(len(dev_buzzes), dev_buzzes_dir))

    expo_buzzes = trainer.test(expo_iter)
    expo_buzzes_dir = 'output/buzzer/neo/expo_buzzes.{}.pkl'.format(model_name)
    with open(expo_buzzes_dir, 'wb') as f:
        pickle.dump(expo_buzzes, f)
    print('Expo buzz {} saved to {}'.format(len(expo_buzzes), expo_buzzes_dir))

    dev_buzzes = trainer.test(dev_iter)
    dev_buzzes_dir = 'output/buzzer/neo/dev_buzzes.{}.pkl'.format(model_name)
    with open(dev_buzzes_dir, 'wb') as f:
        pickle.dump(dev_buzzes, f)
    print('Dev buzz {} saved to {}'.format(len(dev_buzzes), dev_buzzes_dir))

    devtest_buzzes = trainer.test(devtest_iter)
    devtest_buzzes_dir = 'output/buzzer/neo/devtest_buzzes.{}.pkl'.format(model_name)
    with open(devtest_buzzes_dir, 'wb') as f:
        pickle.dump(devtest_buzzes, f)
    print('Devtest buzz {} saved to {}'.format(len(devtest_buzzes), devtest_buzzes_dir))

    report(dev_buzzes_dir)

if __name__ == '__main__':
    main()
