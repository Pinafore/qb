import pickle
import numpy as np
import chainer
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from chainer import Variable
from chainer.backends import cuda
from qanta.datasets.quiz_bowl import QuestionDatabase
from qanta.guesser.abstract import AbstractGuesser
from qanta.util.constants import BUZZER_TRAIN_FOLD, BUZZER_DEV_FOLD

N_GUESSES = 10


def vector_converter_0(guesses_sequence):
    '''default vector converter / feature extractor

    Args:
        guesses_sequence: a sequence (length of question) of list of guesses
            (n_guesses), each entry is (guess, logit, prob)
    Returns:
        a sequence of vectors
    '''
    length = len(guesses_sequence)
    prev_logit_vec = [0. for _ in range(N_GUESSES)]
    prev_prob_vec = [0. for _ in range(N_GUESSES)]
    prev_dict = dict()

    vecs = []
    for i in range(length):
        logit_vec = []
        prob_vec = []
        logit_diff_vec = []
        prob_diff_vec = []
        isnew_vec = []
        guesses = guesses_sequence[i]
        for guess, logit, prob in guesses:
            logit_vec.append(logit)
            prob_vec.append(prob)
            if i > 0 and guess in prev_dict:
                prev_logit, prev_prob = prev_dict[guess]
                logit_diff_vec.append(logit - prev_logit)
                prob_diff_vec.append(prob - prev_prob)
                isnew_vec.append(0)
            else:
                logit_diff_vec.append(logit)
                prob_diff_vec.append(prob)
                isnew_vec.append(1)
        if len(guesses) < N_GUESSES:
            for k in range(max(N_GUESSES - len(guesses), 0)):
                logit_vec.append(0)
                prob_vec.append(0)
                logit_diff_vec.append(0)
                prob_diff_vec.append(0)
                isnew_vec.append(0)
        features = logit_vec[:3] \
            + prob_vec[:3] \
            + isnew_vec[:3] \
            + logit_diff_vec[:3] \
            + prob_diff_vec[:3] \
            + [logit_vec[0] - logit_vec[1], logit_vec[1] - logit_vec[2]] \
            + [prob_vec[0] - prob_vec[1], prob_vec[1] - prob_vec[2]] \
            + [logit_vec[0] - prev_logit_vec[0], logit_vec[1] - prev_logit_vec[1]] \
            + [prob_vec[0] - prev_prob_vec[0], prob_vec[1] - prev_prob_vec[1]] \
            + [sum(isnew_vec[:5])] \
            + [np.average(logit_vec), np.average(prev_logit_vec)] \
            + [np.average(prob_vec), np.average(prev_prob_vec)] \
            + [np.average(logit_vec[:6]), np.average(prev_logit_vec[:5])] \
            + [np.average(prob_vec[:6]), np.average(prev_prob_vec[:5])] \
            + [np.var(logit_vec), np.var(prev_logit_vec)] \
            + [np.var(prob_vec), np.var(prev_prob_vec)] \
            + [np.var(logit_vec[:5]), np.var(prev_logit_vec[:5])] \
            + [np.var(prob_vec[:5]), np.var(prev_prob_vec[:5])]
        vecs.append(np.array(features, dtype=np.float32))
        prev_logit_vec = logit_vec
        prev_prob_vec = prob_vec
        prev_dict = {x: (y, z) for x, y, z in guesses}
    return vecs


def process_question(questions, vector_converter, item):
    qid, q_group = item
    answer = questions[qid].page
    q_group = sorted(q_group.items(), key=lambda x: x[0])
    word_positions, guesses = list(map(list, zip(*q_group)))
    # each entry is a list of (guess, logit, prob) sorted by logit
    labels = np.array([int(g[0][0] == answer) for g in guesses], dtype=np.int32)
    vectors = vector_converter(guesses)
    return qid, vectors, labels, word_positions


def read_data(
        guesser_module='qanta.guesser.dan',
        guesser_class='DanGuesser',
        guesser_config_num=0,
        vector_converter=vector_converter_0):
    guesser_directory = AbstractGuesser.output_path(
        guesser_module, guesser_class, guesser_config_num, '')
    questions = QuestionDatabase().all_questions()
    datasets = []
    # for fold in [BUZZER_TRAIN_FOLD, BUZZER_DEV_FOLD]:
    for fold in [BUZZER_DEV_FOLD, BUZZER_DEV_FOLD]:
        output_path = AbstractGuesser.guess_path(guesser_directory, fold)
        with open(output_path, 'rb') as f:
            df = pickle.load(f)
        pool = Pool(8)
        worker = partial(process_question, questions, vector_converter)
        datasets.append(pool.map(worker, df.items()))
    return datasets


def convert_seq(batch, device=None):
    def to_device_batch(batch):
        if device is None:
            return batch
        elif device < 0:
            return [chainer.dataset.to_device(device, x) for x in batch]
        else:
            xp = cuda.cupy.get_array_module(*batch)
            concat = xp.concatenate(batch, axis=0)
            sections = np.cumsum([len(x) for x in batch[:-1]], dtype=np.int32)
            concat_dev = chainer.dataset.to_device(device, concat)
            batch_dev = cuda.cupy.split(concat_dev, sections)
            return batch_dev
    qids, vectors, labels, positions = list(map(list, zip(*batch)))
    xs = [Variable(x) for x in to_device_batch(vectors)]
    ys = to_device_batch(labels)
    return {'xs': xs, 'ys': ys}


if __name__ == '__main__':
    train, valid = read_data()
