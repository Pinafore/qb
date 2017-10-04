import os
import sys
import random
import numpy as np
import pickle
from collections import defaultdict, namedtuple
from typing import List, Dict, Tuple, Optional
from qanta.config import conf
from qanta.buzzer.util import GUESSERS
from qanta.buzzer import constants as bc
from qanta.util.multiprocess import _multiprocess
from qanta import logging

Batch = namedtuple('Batch', ['qids', 'answers', 'mask', 'vecs', 'results'])

N_GUESSERS = len(GUESSERS)
N_GUESSES = conf['buzzer']['n_guesses']

log = logging.get(__name__)

random.seed(1234)

def dense_vector(dicts: List[List[Dict[str, float]]],
        step_size=1) -> List[List[float]]:

    length = len(dicts)
    prev_vec = [0.02 for _ in range(N_GUESSERS * N_GUESSES)]
    vecs = []
    for i in range(length):
        if len(dicts[i]) != N_GUESSERS:
            raise ValueError("Inconsistent number of guessers ({0}, {1}).".format(
                N_GUESSERS, len(dicts)))
        vec = []
        diff_vec = []
        isnew_vec = []
        for j in range(N_GUESSERS):
            dic = sorted(dicts[i][j].items(), key=lambda x: x[1], reverse=True)
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
        features = [vec[0], vec[1], vec[2],
                    np.average(vec[:10]), np.average(prev_vec[:10]),
                    np.var(vec[:10]), np.var(prev_vec[:10]),
                    sum(isnew_vec[:10]),
                    isnew_vec[0], isnew_vec[1], isnew_vec[2],
                    diff_vec[0], diff_vec[1],
                    vec[0] - vec[1], vec[1] - vec[2], 
                    vec[0] / vec[1], vec[0] / prev_vec[0],
                    vec[0] - prev_vec[0], vec[1] - prev_vec[1]
                    ]

        vecs.append(features)
        prev_vec = vec
    return vecs

class QuestionIterator(object):
    '''Each batch contains:
        qids: list, (batch_size,)
        answers: list, (batch_size,)
        mask: list, (length, batch_size,)
        vecs: xp.float32, (length, batch_size, 4 * NUM_GUESSES)
        results: xp.int32, (length, batch_size)
    '''

    def __init__(self, dataset: list, option2id: Dict[str, int], batch_size:int,
            make_vector=dense_vector, bucket_size=4, step_size=1, neg_weight=1, shuffle=True):
        self.dataset = dataset
        self.option2id = option2id
        self.batch_size = batch_size
        self.make_vector = make_vector
        self.bucket_size = bucket_size
        self.step_size = step_size
        self.neg_weight = neg_weight
        self.shuffle = shuffle
        self.epoch = 0
        self.iteration = 0
        self.batch_index = 0
        self.is_end_epoch = False
        sys.stdout.flush()
        log.info('Creating batches')
        self.create_batches()
        # log.info('Finish creating batches {}'.format(self.n_input))

    def _process_example(self, qid, answer, dicts, results):
        
        results = np.asarray(results, dtype=np.int32)
        length, n_guessers = results.shape

        if n_guessers != N_GUESSERS:
            raise ValueError(
                "Inconsistent number of guessers ({0}, {1}.".format(
                    N_GUESSERS, n_guessers))

        # append the not buzzing action to each time step
        # not buzzing = 1 when no guesser is correct
        new_results = []
        for i in range(length):
            not_buzz = int(not any(results[i] == 1)) * self.neg_weight
            new_results.append(np.append(results[i], not_buzz))
        results = np.asarray(new_results, dtype=np.int32)

        if len(dicts) != length:
            raise ValueError("Inconsistant shape of results and vecs.")
        vecs = self.make_vector(dicts, self.step_size)
        vecs = np.asarray(vecs, dtype=np.float32)
        assert length == vecs.shape[0]
        self.n_input = len(vecs[0])

        padded_length = -((-length) // self.bucket_size) * self.bucket_size
        vecs_padded = np.zeros((padded_length, self.n_input))
        vecs_padded[:length,:self.n_input] = vecs

        results_padded = np.zeros((padded_length, (N_GUESSERS + 1)))
        results_padded[:length, :(N_GUESSERS + 1)] = results

        mask = [1 for _ in range(length)] + \
               [0 for _ in range(padded_length - length)]

        example = (qid, answer, mask, vecs_padded, results_padded)
        return example, padded_length

    def create_batches(self):
        self.batches = []
        buckets = defaultdict(list)
        total = len(self.dataset)
        returns = _multiprocess(self._process_example, self.dataset,
                info="creat batches", multi=False)
        for example, padded_length in returns:
            buckets[padded_length].append(example)

        for examples in buckets.values():
            for i in range(0, len(examples), self.batch_size):
                qids, answers, mask, vecs, results = \
                        zip(*examples[i : i + self.batch_size])
                batch = Batch(qids, answers, mask, vecs, results)
                self.batches.append(batch)

    @property
    def size(self):
        return len(self.batches)
    
    def finalize(self, reset=False):
        if self.shuffle:
            random.shuffle(self.batches)
        if reset:
            self.epoch = 0
            self.iteration = 0
            self.batch_index = 0

    def next_batch(self, xp, train=True):
        self.iteration += 1
        if self.batch_index == 0:
            self.epoch += 1
        self.is_end_epoch = (self.batch_index == self.size - 1)
        qids, answers, mask, vecs, results = self.batches[self.batch_index]

        vecs = xp.asarray(vecs, dtype=xp.float32).swapaxes(0, 1) # length * batch_size * dim
        results = xp.asarray(results, dtype=xp.int32).swapaxes(0, 1) # length * batch_size * n_guessers
        mask = xp.asarray(mask, dtype=xp.float32).T # length * batch_size
        # results = results * 2 - 1 # convert from (0, 1) to (-1, 1)

        self.batch_index = (self.batch_index + 1) % self.size
        batch = Batch(qids, answers, mask, vecs, results)
        return batch
    
    @property
    def epoch_detail(self):
        return self.iteration, self.iteration * 1.0 / self.size
