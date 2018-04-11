import os
import importlib
import warnings
import random
from collections import defaultdict, namedtuple
from abc import ABCMeta, abstractmethod
from typing import List, Dict, Tuple, Optional, NamedTuple
import pickle

import matplotlib
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import numpy as np
from functional import seq

from qanta.reporting.report_generator import ReportGenerator
from qanta.datasets.abstract import TrainingData, QuestionText, Answer
from qanta.datasets.quiz_bowl import QuizBowlDataset, QuestionDatabase
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
    def guess(self, questions: List[QuestionText], max_n_guesses: Optional[int]) -> List[List[Tuple[Answer, float]]]:
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

    def generate_guesses(self, max_n_guesses: int, folds: List[str], word_skip=-1) -> pd.DataFrame:
        """
        Generates guesses for this guesser for all questions in specified folds and returns it as a
        DataFrame

        WARNING: this method assumes that the guesser has been loaded with load or trained with
        train. Unexpected behavior may occur if that is not the case.
        :param max_n_guesses: generate at most this many guesses per question, sentence, and token
        :param folds: which folds to generate guesses for
        :param word_skip: by default, generate sentence level buzzes, if not set to -1 then generate
        buzzes every word_skip words
        :return: dataframe of guesses
        """
        dataset = self.qb_dataset()
        questions_by_fold = dataset.questions_by_fold()

        q_folds = []
        q_qnums = []
        q_sentences = []
        q_tokens = []
        question_texts = []

        for fold in folds:
            questions = questions_by_fold[fold]
            for q in questions:
                for sent, token, text_list in q.partials(word_skip=word_skip):
                    text = ' '.join(text_list)
                    question_texts.append(text)
                    q_folds.append(fold)
                    q_qnums.append(q.qnum)
                    q_sentences.append(sent)
                    q_tokens.append(token)

        guesses_per_question = self.guess(question_texts, max_n_guesses)

        if len(guesses_per_question) != len(question_texts):
            raise ValueError(
                'Guesser has wrong number of answers: len(guesses_per_question)={} len(question_texts)={}'.format(
                    len(guesses_per_question), len(question_texts)))

        log.info('Creating guess dataframe from guesses...')
        df_qnums = []
        df_sentences = []
        df_tokens = []
        df_guesses = []
        df_scores = []
        df_folds = []
        df_guessers = []
        guesser_name = self.display_name()

        for i in range(len(question_texts)):
            guesses_with_scores = guesses_per_question[i]
            fold = q_folds[i]
            qnum = q_qnums[i]
            sentence = q_sentences[i]
            token = q_tokens[i]
            for guess, score in guesses_with_scores:
                df_qnums.append(qnum)
                df_sentences.append(sentence)
                df_tokens.append(token)
                df_guesses.append(guess)
                df_scores.append(score)
                df_folds.append(fold)
                df_guessers.append(guesser_name)

        return pd.DataFrame({
            'qnum': df_qnums,
            'sentence': df_sentences,
            'token': df_tokens,
            'guess': df_guesses,
            'score': df_scores,
            'fold': df_folds,
            'guesser': df_guessers
        })

    @staticmethod
    def guess_path(directory: str, fold: str) -> str:
        return os.path.join(directory, 'guesses_{}.pickle'.format(fold))

    @staticmethod
    def save_guesses(guess_df: pd.DataFrame, directory: str, folds: List[str]):
        for fold in folds:
            log.info('Saving fold {}'.format(fold))
            fold_df = guess_df[guess_df.fold == fold]
            output_path = AbstractGuesser.guess_path(directory, fold)
            fold_df.to_pickle(output_path)

    @staticmethod
    def load_guesses(directory: str, folds=c.GUESSER_GENERATION_FOLDS) -> pd.DataFrame:
        """
        Loads all the guesses pertaining to a guesser inferred from directory
        :param directory: where to load guesses from
        :param folds: folds to load, by default all of them
        :return: guesses across all folds for given directory
        """
        assert len(folds) > 0
        guess_df = None
        for fold in folds:
            input_path = AbstractGuesser.guess_path(directory, fold)
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
        dev_guesses = AbstractGuesser.load_guesses(directory, folds=[c.GUESSER_DEV_FOLD])

        qdb = QuestionDatabase()
        questions = qdb.all_questions()

        # Compute recall and accuracy
        dev_recall = compute_fold_recall(dev_guesses, questions)
        dev_questions = {qnum: q for qnum, q in questions.items() if q.fold == c.GUESSER_DEV_FOLD}
        dev_recall_stats = compute_recall_at_positions(dev_recall)
        dev_summary_accuracy = compute_summary_accuracy(dev_questions, dev_recall_stats)
        dev_summary_recall = compute_summary_recall(dev_questions, dev_recall_stats)

        report_to_kuro(params['kuro_trial_id'] if 'kuro_trial_id' in params else None, dev_summary_accuracy)

        accuracy_plot('/tmp/dev_accuracy.png', dev_summary_accuracy, 'Guesser Dev')
        recall_plot('/tmp/dev_recall.png', dev_questions, dev_summary_recall, 'Guesser Dev')

        # Obtain metrics on number of answerable questions based on the dataset requested
        all_answers = {g for g in qdb.all_answers().values()}
        all_questions = list(qdb.all_questions().values())
        answer_lookup = {qnum: guess for qnum, guess in qdb.all_answers().items()}
        dataset = self.qb_dataset()
        training_data = dataset.training_data()

        min_n_answers = {g for g in training_data[1]}

        train_questions = [q for q in all_questions if q.fold == c.GUESSER_TRAIN_FOLD]
        train_answers = {q.page for q in train_questions}

        dev_questions = [q for q in all_questions if q.fold == c.GUESSER_DEV_FOLD]
        dev_answers = {q.page for q in dev_questions}

        min_n_train_questions = [q for q in train_questions if q.page in min_n_answers]

        all_common_train_dev = train_answers.intersection(dev_answers)
        min_common_train_dev = min_n_answers.intersection(dev_answers)

        all_train_answerable_questions = [q for q in train_questions if q.page in train_answers]
        all_dev_answerable_questions = [q for q in dev_questions if q.page in train_answers]

        min_train_answerable_questions = [q for q in train_questions if q.page in min_n_answers]
        min_dev_answerable_questions = [q for q in dev_questions if q.page in min_n_answers]

        # The next section of code generates the percent of questions correct by the number
        # of training examples.
        Row = namedtuple('Row', [
            'fold', 'guess', 'guesser',
            'qnum', 'score', 'sentence', 'token',
            'correct', 'answerable_1', 'answerable_2',
            'n_examples'
        ])

        train_example_count_lookup = seq(train_questions) \
            .group_by(lambda q: q.page) \
            .smap(lambda page, group: (page, len(group))) \
            .dict()

        def guess_to_row(*args):
            guess = args[1]
            qnum = args[3]
            answer = answer_lookup[qnum]

            return Row(
                *args,
                answer == guess,
                answer in train_answers,
                answer in min_n_answers,
                train_example_count_lookup[answer] if answer in train_example_count_lookup else 0
            )

        dev_data = seq(dev_guesses) \
            .smap(guess_to_row) \
            .group_by(lambda r: (r.qnum, r.sentence)) \
            .smap(lambda key, group: seq(group).max_by(lambda q: q.sentence)) \
            .to_pandas(columns=Row._fields)
        dev_data['correct_int'] = dev_data['correct'].astype(int)
        dev_data['ones'] = 1
        dev_counts = dev_data\
            .groupby('n_examples')\
            .agg({'correct_int': np.mean, 'ones': np.sum})\
            .reset_index()
        correct_by_n_count_plot('/tmp/dev_correct_by_count.png', dev_counts, 'Guesser Dev')
        n_train_vs_fold_plot('/tmp/n_train_vs_dev.png', dev_counts, 'Guesser Dev')

        with open(os.path.join(directory, 'guesser_report.pickle'), 'wb') as f:
            pickle.dump({
                'dev_accuracy': dev_summary_accuracy,
                'guesser_name': self.display_name(),
                'guesser_params': params,
                'directory': directory
            }, f)

        md_output = safe_path(os.path.join(directory, 'guesser_report.md'))
        pdf_output = safe_path(os.path.join(directory, 'guesser_report.pdf'))
        report = ReportGenerator('guesser.md')
        report.create({
            'dev_recall_plot': '/tmp/dev_recall.png',
            'dev_accuracy_plot': '/tmp/dev_accuracy.png',
            'dev_accuracy': dev_summary_accuracy,
            'guesser_name': self.display_name(),
            'guesser_params': params,
            'n_answers_all_folds': len(all_answers),
            'n_total_train_questions': len(train_questions),
            'n_train_questions': len(min_n_train_questions),
            'n_dev_questions': len(dev_questions),
            'n_total_train_answers': len(train_answers),
            'n_train_answers': len(min_n_answers),
            'n_dev_answers': len(dev_answers),
            'all_n_common_train_dev': len(all_common_train_dev),
            'all_p_common_train_dev': len(all_common_train_dev) / max(1, len(dev_answers)),
            'min_n_common_train_dev': len(min_common_train_dev),
            'min_p_common_train_dev': len(min_common_train_dev) / max(1, len(dev_answers)),
            'all_n_answerable_train': len(all_train_answerable_questions),
            'all_p_answerable_train': len(all_train_answerable_questions) / len(train_questions),
            'all_n_answerable_dev': len(all_dev_answerable_questions),
            'all_p_answerable_dev': len(all_dev_answerable_questions) / len(dev_questions),
            'min_n_answerable_train': len(min_train_answerable_questions),
            'min_p_answerable_train': len(min_train_answerable_questions) / len(train_questions),
            'min_n_answerable_dev': len(min_dev_answerable_questions),
            'min_p_answerable_dev': len(min_dev_answerable_questions) / len(dev_questions),
            'dev_correct_by_count_plot': '/tmp/dev_correct_by_count.png',
            'n_train_vs_dev_plot': '/tmp/n_train_vs_dev.png',
        }, md_output, pdf_output)

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

QuestionRecall = namedtuple('QuestionRecall', ['start', 'p_25', 'p_50', 'p_75', 'end'])


def report_to_kuro(kuro_trial_id, summary_accuracy):
    if kuro_trial_id is not None:
        try:
            from kuro.client import Trial
            trial = Trial.from_trial_id(kuro_trial_id)
            trial.report_metric('dev_acc_start', summary_accuracy['start'])
            trial.report_metric('dev_acc_25', summary_accuracy['p_25'])
            trial.report_metric('dev_acc_50', summary_accuracy['p_50'])
            trial.report_metric('dev_acc_75', summary_accuracy['p_75'])
            trial.report_metric('dev_acc_end', summary_accuracy['end'])
            trial.end()
            log.info('Logged guesser accuracies to kuro and ended trial')
        except:
            pass


def question_recall(guesses, qst, question_lookup):
    qnum, sentence, token = qst
    answer = question_lookup[qnum].page
    sorted_guesses = sorted(guesses, reverse=True, key=lambda g: g.score)
    for i, guess_row in enumerate(sorted_guesses, 1):
        if answer == guess_row.guess:
            return qnum, sentence, token, i
    return qnum, sentence, token, None


def compute_fold_recall(guess_df, questions):
    return seq(guess_df)\
        .smap(Guess)\
        .group_by(lambda g: (g.qnum, g.sentence, g.token))\
        .smap(lambda qst, guesses: question_recall(guesses, qst, questions))\
        .group_by(lambda x: x[0])\
        .dict()


def start_of_question(group):
    return seq(group).min_by(lambda g: g[1])[3]


def make_percent_of_question(percent):
    def percent_of_question(group):
        n_sentences = len(group)
        middle = max(1, round(n_sentences * percent))
        middle_element = seq(group).filter(lambda g: g[1] == middle).head_option()
        if middle_element is None:
            return None
        else:
            return middle_element[3]
    return percent_of_question


def end_of_question(group):
    return seq(group).max_by(lambda g: g[1])[3]

percent_25_of_question = make_percent_of_question(.25)
percent_50_of_question = make_percent_of_question(.5)
percent_75_of_question = make_percent_of_question(.75)


def compute_recall_at_positions(recall_lookup):
    recall_stats = {}
    for q in recall_lookup:
        g = recall_lookup[q]
        start = start_of_question(g)
        p_25 = percent_25_of_question(g)
        p_50 = percent_50_of_question(g)
        p_75 = percent_75_of_question(g)
        end = end_of_question(g)
        recall_stats[q] = QuestionRecall(start, p_25, p_50, p_75, end)
    return recall_stats


def compute_summary_accuracy(questions, recall_stats):
    accuracy_stats = {
        'start': 0,
        'p_25': 0,
        'p_50': 0,
        'p_75': 0,
        'end': 0
    }
    n_questions = len(questions)
    for q in questions:
        if q in recall_stats:
            if recall_stats[q].start == 1:
                accuracy_stats['start'] += 1
            if recall_stats[q].p_25 == 1:
                accuracy_stats['p_25'] += 1
            if recall_stats[q].p_50 == 1:
                accuracy_stats['p_50'] += 1
            if recall_stats[q].p_75 == 1:
                accuracy_stats['p_75'] += 1
            if recall_stats[q].end == 1:
                accuracy_stats['end'] += 1

    accuracy_stats['start'] /= n_questions
    accuracy_stats['p_25'] /= n_questions
    accuracy_stats['p_50'] /= n_questions
    accuracy_stats['p_75'] /= n_questions
    accuracy_stats['end'] /= n_questions
    return accuracy_stats


def compute_summary_recall(questions, recall_stats):
    recall_numbers = {
        'start': [],
        'p_25': [],
        'p_50': [],
        'p_75': [],
        'end': []
    }
    for q in questions:
        if q in recall_stats:
            if recall_stats[q].start is not None:
                recall_numbers['start'].append(recall_stats[q].start)
            if recall_stats[q].p_25 is not None:
                recall_numbers['p_25'].append(recall_stats[q].p_25)
            if recall_stats[q].p_50 is not None:
                recall_numbers['p_50'].append(recall_stats[q].p_50)
            if recall_stats[q].p_75 is not None:
                recall_numbers['p_75'].append(recall_stats[q].p_75)
            if recall_stats[q].end is not None:
                recall_numbers['end'].append(recall_stats[q].end)

    return recall_numbers


def compute_recall_plot_data(recall_positions, n_questions,
                             max_recall=conf['n_guesses'] + int(conf['n_guesses'] * .1)):
    """
    Compute the recall, compute recall out a little further than number of guesses to give the
    plot that uses this data some margin on the right side
    """
    x = list(range(1, max_recall + 1))
    y = [0] * max_recall
    for r in recall_positions:
        y[r - 1] += 1
    y = np.cumsum(y) / n_questions
    return x, y


def recall_plot(output, questions, summary_recall, fold_name):
    data = []
    for position, recall_positions in summary_recall.items():
        x_data, y_data = compute_recall_plot_data(recall_positions, len(questions))
        for x, y in zip(x_data, y_data):
            data.append({'x': x, 'y': y, 'position': position})
    data = pd.DataFrame(data)
    g = sb.FacetGrid(data=data, hue='position', size=5, aspect=1.5)
    g.map(plt.plot, 'x', 'y')
    g.add_legend()
    plt.xlabel('Number of Guesses')
    plt.ylabel('Recall')
    plt.subplots_adjust(top=.9)
    g.fig.suptitle('Guesser Recall Through Question on {}'.format(fold_name))
    plt.savefig(output, dpi=200, format='png')
    plt.clf()
    plt.cla()
    plt.close()


def accuracy_plot(output, summary_accuracy, fold_name):
    pd.DataFrame([
        ('start', summary_accuracy['start']),
        ('25%', summary_accuracy['p_25']),
        ('50%', summary_accuracy['p_50']),
        ('75%', summary_accuracy['p_75']),
        ('end', summary_accuracy['end'])],
        columns=['Position', 'Accuracy']
    ).plot.bar('Position', 'Accuracy', title='Accuracy by Position on {}'.format(fold_name))
    plt.savefig(output, dpi=200, format='png')
    plt.clf()
    plt.cla()
    plt.close()


def correct_by_n_count_plot(output, counts, fold):
    counts.plot('n_examples', 'correct_int')
    plt.title('{} fold'.format(fold))
    plt.xlabel('Number of Training Examples')
    plt.ylabel('Percent Correct')
    plt.savefig(output, dpi=200, format='png')
    plt.clf()
    plt.cla()
    plt.close()


def n_train_vs_fold_plot(output, counts, fold):
    counts.plot('n_examples', 'ones')
    plt.title('{} fold'.format(fold))
    plt.xlabel('Number of Training Examples')
    plt.ylabel('Number of {} Examples'.format(fold))
    plt.savefig(output, dpi=200, format='png')
    plt.clf()
    plt.cla()
    plt.close()


def n_guesser_report(report_path, fold, n_samples=10):
    qdb = QuestionDatabase()
    question_lookup = qdb.all_questions()
    questions = [q for q in question_lookup.values() if q.fold == fold]
    guess_dataframes = []
    folds = [fold]
    for g_spec in AbstractGuesser.list_enabled_guessers():
        path = AbstractGuesser.output_path(g_spec.guesser_module, g_spec.guesser_class, '')
        guess_dataframes.append(AbstractGuesser.load_guesses(path, folds=folds))
    df = pd.concat(guess_dataframes)  # type: pd.DataFrame
    guessers = set(df['guesser'].unique())
    n_guessers = len(guessers)
    guesses = []
    for name, group in df.groupby(['guesser', 'qnum', 'sentence', 'token']):
        top_guess = group.sort_values('score', ascending=False).iloc[0]
        guesses.append(top_guess)

    top_df = pd.DataFrame.from_records(guesses)

    guess_lookup = {}
    for name, group in top_df.groupby(['qnum', 'sentence', 'token']):
        guess_lookup[name] = group

    performance = {}
    question_positions = {}
    n_correct_samples = defaultdict(list)
    for q in questions:
        page = q.page
        positions = [(sent, token) for sent, token, _ in q.partials()]
        # Since partials() passes word_skip=-1 each entry is guaranteed to be a sentence
        n_sentences = len(positions)
        q_positions = {
            'start': 1,
            'p_25': max(1, round(n_sentences * .25)),
            'p_50': max(1, round(n_sentences * .5)),
            'p_75': max(1, round(n_sentences * .75)),
            'end': len(positions)
        }
        question_positions[q.qnum] = q_positions
        for sent, token in positions:
            key = (q.qnum, sent, token)
            if key in guess_lookup:
                guesses = guess_lookup[key]
                n_correct = (guesses.guess == page).sum()
                n_correct_samples[n_correct].append(key)
                if n_correct == 0:
                    correct_guessers = 'None'
                elif n_correct == n_guessers:
                    correct_guessers = 'All'
                else:
                    correct_guessers = '/'.join(sorted(guesses[guesses.guess == page].guesser.values))
            else:
                n_correct = 0
                correct_guessers = 'None'
            performance[key] = (n_correct, correct_guessers)

    start_accuracies = []
    p_25_accuracies = []
    p_50_accuracies = []
    p_75_accuracies = []
    end_accuracies = []

    for q in questions:
        qnum = q.qnum
        start_pos = question_positions[qnum]['start']
        p_25_pos = question_positions[qnum]['p_25']
        p_50_pos = question_positions[qnum]['p_50']
        p_75_pos = question_positions[qnum]['p_75']
        end_pos = question_positions[qnum]['end']

        start_accuracies.append((*performance[(qnum, start_pos, 0)], 'start'))
        p_25_accuracies.append((*performance[(qnum, p_25_pos, 0)], 'p_25'))
        p_50_accuracies.append((*performance[(qnum, p_50_pos, 0)], 'p_50'))
        p_75_accuracies.append((*performance[(qnum, p_75_pos, 0)], 'p_75'))
        end_accuracies.append((*performance[(qnum, end_pos, 0)], 'end'))

    all_accuracies = start_accuracies + p_25_accuracies + p_50_accuracies + p_75_accuracies + end_accuracies

    perf_df = pd.DataFrame.from_records(all_accuracies, columns=['n_guessers_correct', 'correct_guessers', 'position'])
    perf_df['count'] = 1
    n_questions = len(questions)

    aggregate_df = (
        perf_df.groupby(['position', 'n_guessers_correct', 'correct_guessers']).count() / n_questions
    ).reset_index()

    fig, ax = plt.subplots(figsize=(12, 8), nrows=2, ncols=3, sharey=True, sharex=True)

    positions = {
        'start': (0, 0),
        'p_25': (0, 1),
        'p_50': (1, 0),
        'p_75': (1, 1),
        'end': (1, 2)
    }

    position_labels = {
        'start': 'Start',
        'p_25': '25%',
        'p_50': '50%',
        'p_75': '75%',
        'end': '100%'
    }
    ax[(0, 2)].axis('off')

    for p, key in positions.items():
        data = aggregate_df[aggregate_df.position == p].pivot(
            index='n_guessers_correct',
            columns='correct_guessers'
        ).fillna(0)['count']
        plot_ax = ax[key]
        data.plot.bar(stacked=True, ax=plot_ax, title='Question Position: {}'.format(position_labels[p]))
        handles, labels = plot_ax.get_legend_handles_labels()
        ax_legend = plot_ax.legend()
        ax_legend.set_visible(False)
        plot_ax.set(xlabel='Number of Correct Guessers', ylabel='Accuracy')

    for plot_ax in list(ax.flatten()):
        for tk in plot_ax.get_yticklabels():
            tk.set_visible(True)
        for tk in plot_ax.get_xticklabels():
            tk.set_rotation('horizontal')
    fig.legend(handles, labels, bbox_to_anchor=(.8, .75))
    fig.suptitle('Accuracy Breakdown by Guesser')
    accuracy_by_n_correct_plot_path = '/tmp/accuracy_by_n_correct_{}.png'.format(fold)
    fig.savefig(accuracy_by_n_correct_plot_path, dpi=200)

    sampled_questions_by_correct = sample_n_guesser_correct_questions(
        question_lookup, guess_lookup, n_correct_samples, n_samples=n_samples
    )

    report = ReportGenerator('compare_guessers.md')
    report.create({
        'dev_accuracy_by_n_correct_plot': accuracy_by_n_correct_plot_path,
        'sampled_questions_by_correct': sampled_questions_by_correct
    }, None, safe_path(report_path))


def sample_n_guesser_correct_questions(question_lookup, guess_lookup, n_correct_samples, n_samples=10):
    sampled_questions_by_correct = defaultdict(list)
    dataset = QuizBowlDataset(guesser_train=True)
    training_data = dataset.training_data()
    answer_counts = defaultdict(int)
    for ans in training_data[1]:
        answer_counts[ans] += 1

    for n_correct, keys in n_correct_samples.items():
        samples = random.sample(keys, min(n_samples, len(keys)))
        for key in samples:
            qnum, sent, token = key
            page = question_lookup[qnum].page
            text = question_lookup[qnum].get_text(sent, token)
            guesses = guess_lookup[key]
            correct_guessers = tuple(guesses[guesses.guess == page].guesser)
            wrong_guessers = tuple(guesses[guesses.guess != page].guesser)
            sampled_questions_by_correct[n_correct].append(
                (text, key, page, answer_counts[page], correct_guessers, wrong_guessers)
            )

    return sampled_questions_by_correct
