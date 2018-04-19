import os
import pickle
import numpy as np
import pandas as pd
import chainer
from tqdm import tqdm
from chainer.backends import cuda
from qanta.datasets.quiz_bowl import QuestionDatabase
from qanta.guesser.abstract import AbstractGuesser
from qanta.util.constants import BUZZER_TRAIN_FOLD, BUZZER_DEV_FOLD


def read_data(
        guesser_module='qanta.guesser.dan',
        guesser_class='DanGuesser',
        guesser_config_num=0):
    guesser_directory = AbstractGuesser.output_path(
        guesser_module, guesser_class, guesser_config_num, '')
    questions = QuestionDatabase().all_questions()
    datasets = []
    # for fold in [BUZZER_TRAIN_FOLD, BUZZER_DEV_FOLD]:
    for fold in [BUZZER_DEV_FOLD, BUZZER_DEV_FOLD]:
        datasets.append([])
        output_path = AbstractGuesser.guess_path(guesser_directory, fold)
        '''
        df = pd.read_pickle(output_path)
        # group by qnum, sort by word position, then list of vectors and labels
        df_grouped = df.groupby('qnum')
        for qid, q_group in df_grouped.groups.items():
            vectors = []
            labels = []
            answer = questions[qid].page
            index = q_group.word_position.sort_values()
            for _, row in q_group.loc[index.index].iterrows():
                vectors.append(row.probs)
                labels.append(row.guess == answer)
            datasets[-1].append((qid, vectors, labels))
        '''
        with open(output_path, 'rb') as f:
            df = pickle.load(f)
        for qid, q_group in tqdm(df.items()):
            vectors = []
            labels = []
            answer = questions[qid].page
            q_group = sorted(q_group.items(), key=lambda x: x[0])
            for _, (probs, guess) in q_group:
                vectors.append(probs)
                labels.append(guess == answer)
            vectors = np.asarray(vectors, dtype=np.float32)
            labels = np.asarray(labels, dtype=np.int32)
            datasets[-1].append((qid, vectors, labels))
    return datasets


def vector_converter(vectors, labels):
    yield vectors, labels


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

    return {'xs': to_device_batch([x for _, x, _ in batch]),
            'ys': to_device_batch([y for _, _, y in batch])}

if __name__ == '__main__':
    train, valid = read_data()
