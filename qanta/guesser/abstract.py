import os
import importlib
import warnings
from collections import defaultdict, namedtuple
from abc import ABCMeta, abstractmethod
from typing import List, Dict, Tuple, Optional, NamedTuple
import pickle

import matplotlib
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    matplotlib.use('Agg')
import pandas as pd
import numpy as np

from qanta.datasets.abstract import TrainingData, QuestionText, Page
from qanta.datasets.quiz_bowl import QuizBowlDataset, QantaDatabase
from qanta.config import conf
from qanta.util import constants as c
from qanta.util.io import safe_path, safe_open
from qanta import qlogging


log = qlogging.get(__name__)


def get_class(instance_module: str, instance_class: str):
    py_instance_module = importlib.import_module(instance_module)
    py_instance_class = getattr(py_instance_module, instance_class)
    return py_instance_class


GuesserSpec = NamedTuple('GuesserSpec', [
    ('dependency_module', Optional[str]),
    ('dependency_class', Optional[str]),
    ('guesser_module', str),
    ('guesser_class', str),
    ('config_num', Optional[int])
])

Guess = namedtuple('Guess', 'fold guess guesser qnum score sentence token')


class AbstractGuesser(metaclass=ABCMeta):
    def __init__(self, config_num: Optional[int]):
        """
        Abstract class representing a guesser. All abstract methods must be implemented. Class
        construction should be light and not load data since this is reserved for the
        AbstractGuesser.load method.

        :param config_num: Required parameter saying which configuration of the guesser to use or explicitly not
            requesting one by passing None. If it is None implementors should not read the guesser config, otherwise
            read the appropriate configuration. This is a positional argument to force all implementors to fail fast
            rather than implicitly
        """
        self.config_num = config_num

    def qb_dataset(self) -> QuizBowlDataset:
        return QuizBowlDataset(guesser_train=True)

    @abstractmethod
    def train(self, training_data: TrainingData) -> None:
        """
        Given training data, train this guesser so that it can produce guesses.

        training_data can be seen as a tuple of two elements which are
        (train_x, train_y, properties).
        In this case train_x is a list of question runs. For example, if the answer for a question
        is "Albert Einstein" the runs might be ["This", "This German", "This German physicist", ...]
        train_y is a list of true labels. The questions are strings and the true labels are strings.
        Labels are in canonical form. Questions are not preprocessed in any way. To implement common
        pre-processing refer to the qanta/guesser/preprocessing module.

        properties is either None or a list of dictionaries that contain extra information about
        each training example

        :param training_data: training data in the format described above
        :return: This function does not return anything
        """
        pass

    @abstractmethod
    def guess(self, questions: List[QuestionText], max_n_guesses: Optional[int]) -> List[List[Tuple[Page, float]]]:
        """
        Given a list of questions as text, return n_guesses number of guesses per question. Guesses
        must be returned in canonical form, are returned with a score in which higher is better, and
        must also be returned in sorted order with the best guess (highest score) at the front of
        the list and worst guesses (lowest score) at the bottom.

        It is guaranteed that before AbstractGuesser.guess is called that either
        AbstractGuesser.train is called or AbstractGuesser.load is called.

        :param questions: Questions to guess on
        :param max_n_guesses: Number of guesses to produce per question, if None then return all
        of them if possible
        :return: List of top guesses per question
        """
        pass

    @classmethod
    @abstractmethod
    def targets(cls) -> List[str]:
        """
        List of files located in directory that are produced by the train method and loaded by the
        save method.
        :return: list of written files
        """
        pass

    @classmethod
    def raw_targets(cls) -> List[str]:
        """
        Similar to targets but it does not join a unique directory prefix. The provided paths are
        raw paths to the targets.
        :return: list of written files
        """
        return []

    @classmethod
    def files(cls, directory: str) -> List[str]:
        return [os.path.join(directory, file) for file in cls.targets()] + cls.raw_targets()

    @classmethod
    @abstractmethod
    def load(cls, directory: str):
        """
        Given the directory used for saving this guesser, create a new instance of the guesser, and
        load it for guessing or scoring.

        :param directory: training data for guesser
        :return: Instance of AbstractGuesser ready for calling guess/score
        """
        pass

    @abstractmethod
    def save(self, directory: str) -> None:
        pass

    def display_name(self) -> str:
        """
        Return the display name of this guesser which is used in reporting scripts to identify this
        particular guesser. By default str() on the classname, but can be overriden
        :return: display name of this guesser
        """
        return self.__class__.__name__

    def parameters(self) -> Dict:
        """
        Return the parameters of the model. This is displayed as part of the report to make
        identifying particular runs of particular hyper parameters easier. str(self.parameters())
        will be called at some point to display it as well as making a pickle of parameters.
        :return: model parameters
        """
        return {}

    def generate_guesses(self, max_n_guesses: int, folds: List[str],
                         char_skip=25, full_question=False, first_sentence=False) -> pd.DataFrame:
        """
        Generates guesses for this guesser for all questions in specified folds and returns it as a
        DataFrame

        WARNING: this method assumes that the guesser has been loaded with load or trained with
        train. Unexpected behavior may occur if that is not the case.
        :param max_n_guesses: generate at most this many guesses per question, sentence, and token
        :param folds: which folds to generate guesses for
        :param char_skip: generate guesses every 10 characters
        :return: dataframe of guesses
        """
        if full_question and first_sentence:
            raise ValueError('Invalid option combination')

        dataset = self.qb_dataset()
        questions_by_fold = dataset.questions_by_fold()

        q_folds = []
        q_qnums = []
        q_char_indices = []
        q_proto_ids = []
        question_texts = []

        for fold in folds:
            questions = questions_by_fold[fold]
            for q in questions:
                if full_question:
                    question_texts.append(q.text)
                    q_folds.append(fold)
                    q_qnums.append(q.qanta_id)
                    q_char_indices.append(len(q.text))
                    q_proto_ids.append(q.proto_id)
                elif first_sentence:
                    question_texts.append(q.first_sentence)
                    q_folds.append(fold)
                    q_qnums.append(q.qanta_id)
                    q_char_indices.append(q.tokenizations[0][1])
                    q_proto_ids.append(q.proto_id)
                else:
                    for text_run, char_ix in zip(*q.runs(char_skip)):
                        question_texts.append(text_run)
                        q_folds.append(fold)
                        q_qnums.append(q.qanta_id)
                        q_char_indices.append(char_ix)
                        q_proto_ids.append(q.proto_id)

        guesses_per_question = self.guess(question_texts, max_n_guesses)

        if len(guesses_per_question) != len(question_texts):
            raise ValueError(
                'Guesser has wrong number of answers: len(guesses_per_question)={} len(question_texts)={}'.format(
                    len(guesses_per_question), len(question_texts)))

        log.info('Creating guess dataframe from guesses...')
        df_qnums = []
        df_proto_id = []
        df_char_indices = []
        df_guesses = []
        df_scores = []
        df_folds = []
        df_guessers = []
        guesser_name = self.display_name()

        for i in range(len(question_texts)):
            guesses_with_scores = guesses_per_question[i]
            fold = q_folds[i]
            qnum = q_qnums[i]
            proto_id = q_proto_ids[i]
            char_ix = q_char_indices[i]
            for guess, score in guesses_with_scores:
                df_qnums.append(qnum)
                df_proto_id.append(proto_id)
                df_char_indices.append(char_ix)
                df_guesses.append(guess)
                df_scores.append(score)
                df_folds.append(fold)
                df_guessers.append(guesser_name)

        return pd.DataFrame({
            'qanta_id': df_qnums,
            'proto_id': df_proto_id,
            'char_index': df_char_indices,
            'guess': df_guesses,
            'score': df_scores,
            'fold': df_folds,
            'guesser': df_guessers
        })

    @staticmethod
    def guess_path(directory: str, fold: str, output_type: str) -> str:
        return os.path.join(directory, f'guesses_{output_type}_{fold}.pickle')

    @staticmethod
    def save_guesses(guess_df: pd.DataFrame, directory: str, folds: List[str], output_type):
        for fold in folds:
            log.info('Saving fold {}'.format(fold))
            fold_df = guess_df[guess_df.fold == fold]
            output_path = AbstractGuesser.guess_path(directory, fold, output_type)
            fold_df.to_pickle(output_path)

    @staticmethod
    def load_guesses(directory: str, output_type='char', folds=c.GUESSER_GENERATION_FOLDS) -> pd.DataFrame:
        """
        Loads all the guesses pertaining to a guesser inferred from directory
        :param directory: where to load guesses from
        :param output_type: One of: char, full, first
        :param folds: folds to load, by default all of them
        :return: guesses across all folds for given directory
        """
        assert len(folds) > 0
        guess_df = None
        for fold in folds:
            input_path = AbstractGuesser.guess_path(directory, fold, output_type)
            if guess_df is None:
                guess_df = pd.read_pickle(input_path)
            else:
                new_guesses_df = pd.read_pickle(input_path)
                guess_df = pd.concat([guess_df, new_guesses_df])

        return guess_df

    @staticmethod
    def load_all_guesses(directory_prefix='') -> pd.DataFrame:
        """
        Loads all guesses from all guessers and folds
        :return:
        """
        guess_df = None
        guessers = conf['guessers']
        for guesser_key, g in guessers.items():
            g = guessers[guesser_key]
            if g['enabled']:
                input_path = os.path.join(directory_prefix, c.GUESSER_TARGET_PREFIX, g['class'])
                if guess_df is None:
                    guess_df = AbstractGuesser.load_guesses(input_path)
                else:
                    new_guess_df = AbstractGuesser.load_guesses(input_path)
                    guess_df = pd.concat([guess_df, new_guess_df])

        return guess_df

    @staticmethod
    def load_guess_score_map(guess_df: pd.DataFrame) -> defaultdict:
        guess_score_map = defaultdict(dict)
        for row in guess_df.itertuples():
            guess_score_map[row.guesser][(row.qnum, row.sentence, row.token, row.guess)] = row.score

        return guess_score_map

    def create_report(self, directory: str):
        with open(os.path.join(directory, 'guesser_params.pickle'), 'rb') as f:
            params = pickle.load(f)

        qdb = QantaDatabase()
        guesser_train = qdb.guess_train_questions
        guesser_dev = qdb.guess_dev_questions

        train_pages = {q.page for q in guesser_train}
        dev_pages = {q.page for q in guesser_dev}

        unanswerable_answer_percent = len(dev_pages - train_pages) / len(dev_pages)
        answerable = 0
        for q in guesser_dev:
            if q.page in train_pages:
                answerable += 1
        unanswerable_question_percent = answerable / len(guesser_dev)

        char_guess_df = AbstractGuesser.load_guesses(directory, folds=[c.GUESSER_DEV_FOLD], output_type='char')
        dev_df = pd.DataFrame({
            'page': [q.page for q in guesser_dev],
            'qanta_id': [q.qanta_id for q in guesser_dev],
            'text_length': [len(q.text) for q in guesser_dev]
        })

        char_df = char_guess_df.merge(dev_df, on='qanta_id')
        char_df['correct'] = char_df.guess == char_df.page
        char_df['char_percent'] = (char_df['char_index'] / char_df['text_length']).clip_upper(1.0)

        first_guess_df = AbstractGuesser.load_guesses(directory, folds=[c.GUESSER_DEV_FOLD], output_type='first')

        full_guess_df = AbstractGuesser.load_guesses(directory, folds=[c.GUESSER_DEV_FOLD], output_type='full')

        with open(os.path.join(directory, 'guesser_report.pickle'), 'wb') as f:
            pickle.dump({
                'dev_accuracy': 0,
                'unanswerable_answer_percent': unanswerable_answer_percent,
                'unanswerable_question_percent': unanswerable_question_percent,
                'guesser_name': self.display_name(),
                'guesser_params': params
            }, f)

    @staticmethod
    def list_enabled_guessers() -> List[GuesserSpec]:
        guessers = conf['guessers']
        enabled_guessers = []
        for guesser, configs in guessers.items():
            for config_num, g_conf in enumerate(configs):
                if g_conf['enabled']:
                    dependency = g_conf['luigi_dependency']
                    parts = guesser.split('.')
                    guesser_module = '.'.join(parts[:-1])
                    guesser_class = parts[-1]

                    if dependency is None:
                        dependency_module = None
                        dependency_class = None
                    else:
                        parts = dependency.split('.')
                        dependency_module = '.'.join(parts[:-1])
                        dependency_class = parts[-1]

                    enabled_guessers.append(GuesserSpec(
                        dependency_module, dependency_class, guesser_module, guesser_class, config_num
                    ))

        return enabled_guessers

    @staticmethod
    def output_path(guesser_module: str, guesser_class: str, config_num: int, file: str):
        guesser_path = '{}.{}'.format(guesser_module, guesser_class)
        return safe_path(os.path.join(
            c.GUESSER_TARGET_PREFIX, guesser_path, str(config_num), file
        ))

    @staticmethod
    def reporting_path(guesser_module: str, guesser_class: str, config_num: int, file: str):
        guesser_path = '{}.{}'.format(guesser_module, guesser_class)
        return safe_path(os.path.join(
            c.GUESSER_REPORTING_PREFIX, guesser_path, str(config_num), file
        ))

    def web_api(self, host='0.0.0.0', port=5000, debug=False):
        from flask import Flask, jsonify, request

        app = Flask(__name__)

        @app.route('/api/answer_question', methods=['POST'])
        def answer_question():
            text = request.form['text']
            guess, score = self.guess([text], 1)[0][0]
            return jsonify({'guess': guess, 'score': float(score)})

        app.run(host=host, port=port, debug=debug)

    @staticmethod
    def multi_guesser_web_api(guesser_names: List[str], host='0.0.0.0', port=5000, debug=False):
        from flask import Flask, jsonify, request

        app = Flask(__name__)

        guesser_lookup = {}
        for name, g in conf['guessers'].items():
            g_qualified_name = g['class']
            parts = g_qualified_name.split('.')
            g_module = '.'.join(parts[:-1])
            g_classname = parts[-1]
            guesser_lookup[name] = (get_class(g_module, g_classname), g_qualified_name)

        log.info(f'Loading guessers: {guesser_names}')
        guessers = {}
        for name in guesser_names:
            if name in guesser_lookup:
                g_class, g_qualified_name = guesser_lookup[name]
                guesser_path = os.path.join('output/guesser', g_qualified_name)
                log.info(f'Loading "{name}" corresponding to "{g_qualified_name}" located at "{guesser_path}"')
                guessers[name] = g_class.load(guesser_path)
            else:
                log.info(f'Guesser with name="{name}" not found')

        @app.route('/api/guesser', methods=['POST'])
        def guess():
            if 'guesser_name' not in request.form:
                response = jsonify({'errors': 'Missing expected field "guesser_name"'})
                response.status_code = 400
                return response

            if 'text' not in request.form:
                response = jsonify({'errors': 'Missing expected field "text"'})
                response.status_code = 400
                return response

            g_name = request.form['guesser_name']
            if g_name not in guessers:
                response = jsonify(
                    {'errors': f'Guesser "{g_name}" invalid, options are: "{list(guessers.keys())}"'}
                )
                response.status_code = 400
                return response
            text = request.form['text']
            guess, score = guessers[g_name].guess([text], 1)[0][0]
            return jsonify({'guess': guess, 'score': float(score)})

        app.run(host=host, port=port, debug=debug)
