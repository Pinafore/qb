from typing import List, Tuple, Optional
from pprint import pformat
import tempfile
import pickle
import os
import random
import re
from qanta.datasets.abstract import Answer, TrainingData, QuestionText
from qanta.guesser.abstract import AbstractGuesser
from qanta.util.io import shell, safe_path
from qanta.config import conf
from qanta import qlogging


log = qlogging.get(__name__)


def format_question(text):
    return re.sub(r'[^a-z0-9 ]+', '', text.lower())


class VWGuesser(AbstractGuesser):
    def __init__(self, config_num):
        super().__init__(config_num)
        self.label_to_i = None
        self.i_to_label = None
        self.max_label = None
        self.model_file = None
        if self.config_num is not None:
            guesser_conf = conf['guessers']['qanta.guesser.vw.VWGuesser'][self.config_num]
            self.multiclass_one_against_all = guesser_conf['multiclass_one_against_all']
            self.multiclass_online_trees = guesser_conf['multiclass_online_trees']
            self.l1 = guesser_conf['l1']
            self.l2 = guesser_conf['l2']
            self.passes = guesser_conf['passes']
            self.learning_rate = guesser_conf['learning_rate']
            self.decay_learning_rate = guesser_conf['decay_learning_rate']
            self.bits = guesser_conf['bits']
            self.ngrams = guesser_conf['ngrams']
            self.skips = guesser_conf['skips']
            self.random_seed = guesser_conf['random_seed']
            if not (self.multiclass_one_against_all != self.multiclass_online_trees):
                raise ValueError('The options multiclass_one_against_all and multiclass_online_trees are XOR')

    @classmethod
    def targets(cls) -> List[str]:
        return ['vw_guesser.vw', 'vw_guesser.pickle']

    def parameters(self):
        return {
            'multiclass_one_against_all': self.multiclass_one_against_all,
            'multiclass_online_trees': self.multiclass_online_trees,
            'l1': self.l1,
            'l2': self.l2,
            'passes': self.passes,
            'learning_rate': self.learning_rate,
            'decay_learning_rate': self.decay_learning_rate,
            'bits': self.bits,
            'random_seed': self.random_seed
        }

    def save(self, directory: str) -> None:
        model_path = safe_path(os.path.join(directory, 'vw_guesser.vw'))
        shell(f'mv {self.model_file}.vw {model_path}')
        self.model_file = model_path
        data = {
            'label_to_i': self.label_to_i,
            'i_to_label': self.i_to_label,
            'max_label': self.max_label,
            'multiclass_one_against_all': self.multiclass_one_against_all,
            'multiclass_online_trees': self.multiclass_online_trees,
            'l1': self.l1,
            'l2': self.l2,
            'passes': self.passes,
            'learning_rate': self.learning_rate,
            'decay_learning_rate': self.decay_learning_rate,
            'bits': self.bits,
            'config_num': self.config_num,
            'random_seed': self.random_seed
        }
        data_pickle_path = os.path.join(directory, 'vw_guesser.pickle')
        with open(data_pickle_path, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, directory: str):
        data_pickle_path = os.path.join(directory, 'vw_guesser.pickle')
        with open(data_pickle_path, 'rb') as f:
            data = pickle.load(f)
        guesser = VWGuesser(None)
        guesser.model_file = os.path.join(directory, 'vw_guesser.vw')
        guesser.label_to_i = data['label_to_i']
        guesser.i_to_label = data['i_to_label']
        guesser.max_label = data['max_label']
        guesser.multiclass_one_against_all = data['multiclass_one_against_all']
        guesser.multiclass_online_trees = data['multiclass_online_trees']
        guesser.l1 = data['l1']
        guesser.l2 = data['l2']
        guesser.passes = data['passes']
        guesser.learning_rate = data['learning_rate']
        guesser.decay_learning_rate = data['decay_learning_rate']
        guesser.bits = data['bits']
        guesser.random_seed = data['random_seed']
        return guesser

    def guess(self,
              questions: List[QuestionText],
              max_n_guesses: Optional[int]) -> List[List[Tuple[Answer, float]]]:
        with tempfile.NamedTemporaryFile('w', delete=False) as f:
            file_name = f.name
            for q in questions:
                features = format_question(q)
                f.write(f'1 |words {features}\n')
        shell(f'vw -t -i {self.model_file} -p {file_name}_preds -d {file_name}')
        predictions = []
        with open(f'{file_name}_preds') as f:
            for line in f:
                label = int(line)
                predictions.append([(self.i_to_label[label], 0)])
        shell(f'{file_name}.preds {file_name}')
        return predictions

    def train(self, training_data: TrainingData) -> None:
        log.info(f'Config:\n{pformat(self.parameters())}')
        questions = training_data[0]
        answers = training_data[1]

        x_data = []
        y_data = []
        for q, ans in zip(questions, answers):
            for sent in q:
                x_data.append(sent)
                y_data.append(ans)

        label_set = set(answers)
        self.label_to_i = {label: i for i, label in enumerate(label_set, 1)}
        self.i_to_label = {i: label for label, i in self.label_to_i.items()}
        self.max_label = len(self.label_to_i)

        with tempfile.NamedTemporaryFile('w', delete=False) as f:
            file_name = f.name
            zipped = list(zip(x_data, y_data))
            random.shuffle(zipped)
            for x, y in zipped:
                features = format_question(x)
                label = self.label_to_i[y]
                f.write('{label} |words {features}\n'.format(label=label, features=features))

        if self.multiclass_online_trees:
            multiclass_flag = '--log_multi'
        elif self.multiclass_one_against_all:
            multiclass_flag = '--oaa'
        else:
            raise ValueError('The options multiclass_one_against_all and multiclass_online_trees are XOR')

        with tempfile.NamedTemporaryFile(delete=True) as f:
            self.model_file = f.name

        options = [
            'vw',
            '-k',
            f'{multiclass_flag}',
            f'{self.max_label}',
            f'-d {file_name}',
            f'-f {self.model_file}.vw',
            '--loss_function logistic',
            '-c',
            f'--passes {self.passes}',
            f'-b {self.bits}',
            f'-l {self.learning_rate}',
            f'--decay_learning_rate {self.decay_learning_rate}',
            f'--random_seed {self.random_seed}'
        ]

        for n in self.ngrams:
            options.append(f'--ngram {n}')

        for n in self.skips:
            options.append(f'--skips {n}')

        if self.l1 != 0:
            options.append(f'--l1 {self.l1}')

        if self.l2 != 0:
            options.append(f'--l2 {self.l2}')

        command = ' '.join(options)
        log.info(f'Running:\n{command}')

        try:
            shell(command)
        finally:
            shell(f'rm {file_name} {file_name}.cache')
