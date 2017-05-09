import random
import numpy as np
from collections import defaultdict, namedtuple
from typing import List, Dict, Tuple, Optional
from qanta.buzzer import constants as bc

Batch = namedtuple('Batch', ['qids', 'answers', 'mask', 'vecs', 'results'])

class QuestionIterator(object):
    '''Each batch contains:
        qids: list, (batch_size,)
        answers: list, (batch_size,)
        mask: list, (length, batch_size,)
        vecs: xp.float32, (length, batch_size, 4 * NUM_GUESSES)
        results: xp.int32, (length, batch_size)
    '''

    def __init__(self, dataset: list, option2id: Dict[str, int], batch_size:int,
            bucket_size=4, shuffle=True, only_hopeful=False):
        self.dataset = dataset
        self.option2id = option2id
        self.batch_size = batch_size
        self.bucket_size = bucket_size
        self.shuffle = shuffle
        self.only_hopeful = only_hopeful
        self.epoch = 0
        self.iteration = 0
        self.batch_index = 0
        self.is_end_epoch = False
        self.create_batches()

    def dense_vector(self, dicts: List[List[Dict[str, float]]]) -> List[List[float]]:
        '''Generate dense vectors from a sequence of guess dictionaries.
        dicts: a sequence of guess dictionaries for each guesser
        '''
        length = len(dicts)
        prev_vec = [0. for _ in range(bc.N_GUESSERS * bc.N_GUESSES)]
        vecs = []
        for i in range(length):
            if len(dicts[i]) != bc.N_GUESSERS:
                raise ValueError("Inconsistent number of guessers ({0}, {1}).".format(
                    bc.N_GUESSERS, len(dicts)))
            vec = []
            diff_vec = []
            isnew_vec = []
            for j in range(bc.N_GUESSERS):
                dic = sorted(dicts[i][j].items(), key=lambda x: x[1], reverse=True)
                for guess, score in dic:
                    vec.append(score)
                    if i > 0 and guess in dicts[i-1][j]:
                        diff_vec.append(score - dicts[i-1][j][guess])
                        isnew_vec.append(0)
                    else:
                        diff_vec.append(score) 
                        isnew_vec.append(1)
                if len(dic) < bc.N_GUESSES:
                    for k in range(max(bc.N_GUESSES - len(dic), 0)):
                        vec.append(bc.MIN_SCORE)
                        diff_vec.append(0)
                        isnew_vec.append(0)
            vecs.append(vec + diff_vec + isnew_vec + prev_vec)
            prev_vec = vec
        return vecs

    def create_batches(self):
        bucket_size = self.bucket_size
        self.batches = []
        buckets = defaultdict(list)
        for example in self.dataset:
            # pad the sequence of predictions
            qid, answer, vecs, results = example
            
            results = np.asarray(results, dtype=np.int32)
            length, n_guessers = results.shape

            if n_guessers != bc.N_GUESSERS:
                raise ValueError(
                    "Inconsistent number of guessers ({0}, {1}.".format(
                        bc.N_GUESSERS, len(n_guessers)))

            # hopeful means any guesser guesses correct any time step
            hopeful = np.any(results == 1)
            if self.only_hopeful and not hopeful:
                continue

            # append the not buzzing action to each time step
            # not buzzing = 1 when no guesser is correct
            new_results = []
            for i in range(length):
                not_buzz = int(not any(results[i] == 1)) * bc.NEG_WEIGHT
                new_results.append(np.append(results[i], not_buzz))
            results = np.asarray(new_results, dtype=np.int32)

            if len(vecs) != length:
                raise ValueError("Inconsistant shape of results and vecs.")
            vecs = self.dense_vector(vecs)
            vecs = np.asarray(vecs, dtype=np.float32)
            self.n_input = len(vecs[0])

            padded_length = -((-length) // bucket_size) * bucket_size
            vecs_padded = np.zeros((padded_length, self.n_input))
            vecs_padded[:length,:self.n_input] = vecs

            results_padded = np.zeros((padded_length, (bc.N_GUESSERS + 1)))
            results_padded[:length, :(bc.N_GUESSERS + 1)] = results

            mask = [1 for _ in range(length)] + \
                   [0 for _ in range(padded_length - length)]

            buckets[padded_length].append((qid, answer, mask, vecs_padded,
                results_padded))

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
