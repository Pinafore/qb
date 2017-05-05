import random
import numpy as np
from collections import defaultdict, namedtuple
from multiprocessing import Pool
from functools import partial

from qanta.datasets.quiz_bowl import QuestionDatabase
from qanta.datasets.quiz_bowl import QuizBowlDataset
from qanta.preprocess import format_guess
from qanta.guesser.abstract import AbstractGuesser

Batch = namedtuple('Batch', ['qids', 'answers', 'mask', 'vecs', 'results'])

class QuestionIterator(object):
    '''Each batch contains:
        qids: list, (batch_size,)
        answers: list, (batch_size,)
        mask: list, (length, batch_size,)
        vecs: xp.float32, (length, batch_size, 4 * NUM_GUESSES)
        results: xp.int32, (length, batch_size)
    '''

    def __init__(self, dataset, id2option, batch_size, bucket_size=4, shuffle=True,
            only_hopeful=False):
        self.dataset = dataset
        self.id2option = id2option
        self.batch_size = batch_size
        self.bucket_size = bucket_size
        self.shuffle = shuffle
        self.only_hopeful = only_hopeful
        self.epoch = 0
        self.iteration = 0
        self.batch_index = 0
        self.is_end_epoch = False
        self.create_batches()

    def create_batches(self):
        bucket_size = self.bucket_size
        self.batches = []
        buckets = defaultdict(list)
        for example in self.dataset:
            # pad the sequence of predictions
            qid, answer, vecs, results = example
            length = len(vecs)

            if self.only_hopeful and not any(np.asarray(results) == 1):
                continue

            # add two features
            # for i in range(length):
            #     vecs[i].append(np.var(vecs[i]))
            #     vecs[i].append(i)

            # new_vecs = []

            self.n_input = len(vecs[0])
            padded_length = -((-length) // bucket_size) * bucket_size
            vecs_padded = np.zeros((padded_length, self.n_input))
            vecs_padded[:length,:self.n_input] = vecs
            results += [0 for _ in range(padded_length - length)]
            mask = [1 for _ in range(length)] + \
                   [0 for _ in range(padded_length - length)]
            buckets[padded_length].append((qid, answer, mask, vecs_padded, results))
        for examples in buckets.values():
            for i in range(0, len(examples), self.batch_size):
                qids, answers, mask, vecs, results = zip(*examples[i : i + self.batch_size])
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
        mask = xp.asarray(mask, dtype=xp.float32).T # length * batch_size
        results = xp.asarray(results, dtype=xp.int32).T # length * batch_size
        # results = results * 2 - 1 # convert from (0, 1) to (-1, 1)

        self.batch_index = (self.batch_index + 1) % self.size
        batch = Batch(qids, answers, mask, vecs, results)
        return batch
    
    @property
    def epoch_detail(self):
        return self.iteration, self.iteration * 1.0 / self.size


