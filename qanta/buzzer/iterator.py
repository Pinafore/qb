import random
import numpy as np
from collections import defaultdict, namedtuple

Batch = namedtuple('Batch', ['qids', 'answers', 'mask', 'vecs', 'results'])

class QuestionIterator(object):
    '''Each batch contains:
        qids: list, (batch_size,)
        answers: list, (batch_size,)
        mask: list, (length, batch_size,)
        vecs: xp.float32, (length, batch_size, 4 * NUM_GUESSES)
        results: xp.int32, (length, batch_size)
    '''

    def __init__(self, dataset, option2id, batch_size, bucket_size=4, shuffle=True,
            only_hopeful=False):
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

    def dense_vector(self, dicts):
        num_guesses = len(dicts[0])
        vecs = []
        prev_vec = [0 for _ in range(num_guesses)]
        prev_dict = {}

        for curr_dict in dicts:
            if len(curr_dict) != num_guesses:
                raise ValueError("Inconsistent number of guesses")
            curr_vec = sorted(curr_dict.items(), key=lambda x: x[1])
            diff_vec, isnew_vec = [], []
            for guess, score in curr_vec:
                if guess not in prev_dict:
                    diff_vec.append(score)
                    isnew_vec.append(1)
                else:
                    diff_vec.append(score - prev_dict[guess])
                    isnew_vec.append(0)
            curr_vec = [x[1] for x in curr_vec]
            vec = curr_vec + prev_vec + diff_vec + isnew_vec
            vecs.append(vec)
            prev_vec = curr_vec
            prev_dict = curr_dict
        return vecs

    def sparse_vector(self, dicts):
        vecs = []
        prev_vec = [0 for _ in range(len(self.option2id) + 1)]
        for curr_dict in dicts:
            vec = [0 for _ in range(len(self.option2id) + 1)]
            for guess, score in curr_dict.items():
                guess = self.option2id.get(guess, len(self.option2id))
                vec[guess] = score
            vecs.append(vec + prev_vec)
            prev_vec = vec
        return vecs

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

            vecs = self.sparse_vector(vecs)

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


