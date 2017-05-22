import importlib
import pickle
import time

import luigi
from luigi import LocalTarget, Task, WrapperTask

from qanta.config import conf
from qanta.util import constants as c
from qanta.guesser.abstract import AbstractGuesser, n_guesser_report
from qanta.pipeline.preprocess import DownloadData
from qanta import logging

log = logging.get(__name__)


def get_class(instance_module: str, instance_class: str):
    py_instance_module = importlib.import_module(instance_module)
    py_instance_class = getattr(py_instance_module, instance_class)
    return py_instance_class


class EmptyTask(luigi.Task):
    def complete(self):
        return True


class TrainGuesser(Task):
    guesser_module = luigi.Parameter()  # type: str
    guesser_class = luigi.Parameter()  # type: str
    dependency_module = luigi.Parameter()  # type: str
    dependency_class = luigi.Parameter()  # type: str

    def requires(self):
        yield DownloadData()
        if self.dependency_class is not None and self.dependency_module is not None:
            dependency_class = get_class(self.dependency_module, self.dependency_class)
            yield dependency_class()

    def run(self):
        guesser_class = get_class(self.guesser_module, self.guesser_class)
        guesser_instance = guesser_class()  # type: AbstractGuesser
        qb_dataset = guesser_instance.qb_dataset()
        start_time = time.time()
        guesser_instance.train(qb_dataset.training_data())
        end_time = time.time()
        guesser_instance.save(AbstractGuesser.output_path(self.guesser_module, self.guesser_class, ''))
        params = guesser_instance.parameters()
        params['training_time'] = end_time - start_time
        params_path = AbstractGuesser.output_path(self.guesser_module, self.guesser_class, 'guesser_params.pickle')
        with open(params_path, 'wb') as f:
            pickle.dump(params, f)

    def output(self):
        guesser_class = get_class(self.guesser_module, self.guesser_class)
        guesser_targets = [
            LocalTarget(file)
            for file in guesser_class.files(
                AbstractGuesser.output_path(self.guesser_module, self.guesser_class, '')
            )]

        return [
            LocalTarget(AbstractGuesser.output_path(self.guesser_module, self.guesser_class, '')),
            LocalTarget(
                AbstractGuesser.output_path(self.guesser_module, self.guesser_class, 'guesser_params.pickle'))
        ] + guesser_targets


class GenerateGuesses(Task):
    guesser_module = luigi.Parameter()  # type: str
    guesser_class = luigi.Parameter()  # type: str
    dependency_module = luigi.Parameter()  # type: str
    dependency_class = luigi.Parameter()  # type: str
    n_guesses = luigi.IntParameter(default=conf['n_guesses'])  # type: int
    fold = luigi.Parameter()  # type: str

    def requires(self):
        yield TrainGuesser(
            guesser_module=self.guesser_module,
            guesser_class=self.guesser_class,
            dependency_module=self.dependency_module,
            dependency_class=self.dependency_class
        )

    def run(self):
        guesser_class = get_class(self.guesser_module, self.guesser_class)
        guesser_directory = AbstractGuesser.output_path(self.guesser_module, self.guesser_class, '')
        guesser_instance = guesser_class.load(guesser_directory)  # type: AbstractGuesser

        if self.fold in {c.GUESSER_TRAIN_FOLD, c.GUESSER_DEV_FOLD}:
            word_skip = conf['guesser_word_skip']
        else:
            word_skip = conf['buzzer_word_skip']

        log.info('Generating and saving guesses for {} fold with word_skip={}...'.format(self.fold, word_skip))
        start_time = time.time()
        guess_df = guesser_instance.generate_guesses(self.n_guesses, [self.fold], word_skip=word_skip)
        end_time = time.time()
        log.info('Guessing on {} fold took {}s, saving guesses...'.format(self.fold, end_time - start_time))
        guesser_class.save_guesses(guess_df, guesser_directory, [self.fold])
        log.info('Done saving guesses')

    def output(self):
        return LocalTarget(AbstractGuesser.output_path(
            self.guesser_module, self.guesser_class,
            'guesses_{}.pickle'.format(self.fold)
        ))


class GenerateAllGuesses(WrapperTask):
    def requires(self):
        for g_spec in AbstractGuesser.list_enabled_guessers():
            for fold in c.GUESSER_GENERATION_FOLDS:
                yield GenerateGuesses(
                    guesser_module=g_spec.guesser_module,
                    guesser_class=g_spec.guesser_class,
                    dependency_module=g_spec.dependency_module,
                    dependency_class=g_spec.dependency_class,
                    fold=fold
                )


class GuesserReport(Task):
    guesser_module = luigi.Parameter()  # type: str
    guesser_class = luigi.Parameter()  # type: str
    dependency_module = luigi.Parameter()  # type: str
    dependency_class = luigi.Parameter()  # type: str

    def requires(self):
        yield GenerateGuesses(
            guesser_module=self.guesser_module,
            guesser_class=self.guesser_class,
            dependency_module=self.dependency_module,
            dependency_class=self.dependency_class,
            fold=c.GUESSER_DEV_FOLD
        )

    def run(self):
        guesser_class = get_class(self.guesser_module, self.guesser_class)
        guesser_directory = AbstractGuesser.output_path(self.guesser_module, self.guesser_class, '')
        guesser_instance = guesser_class()
        guesser_instance.create_report(guesser_directory)

    def output(self):
        return [LocalTarget(AbstractGuesser.output_path(
            self.guesser_module,
            self.guesser_class,
            'guesser_report.pdf')
        ), LocalTarget(AbstractGuesser.output_path(
            self.guesser_module,
            self.guesser_class,
            'guesser_report.pickle'
        ))]


class AllSingleGuesserReports(WrapperTask):
    def requires(self):
        for g_spec in AbstractGuesser.list_enabled_guessers():
            yield GuesserReport(
                guesser_module=g_spec.guesser_module,
                guesser_class=g_spec.guesser_class,
                dependency_module=g_spec.dependency_module,
                dependency_class=g_spec.dependency_class
            )


class CompareGuessersReport(Task):
    def requires(self):
        yield AllSingleGuesserReports()

    def run(self):
        n_guesser_report(c.COMPARE_GUESSER_REPORT_PATH.format(c.GUESSER_DEV_FOLD), c.GUESSER_DEV_FOLD)

    def output(self):
        return LocalTarget(c.COMPARE_GUESSER_REPORT_PATH.format(c.GUESSER_DEV_FOLD))


class AllGuesserReports(WrapperTask):
    def requires(self):
        yield AllSingleGuesserReports()
        yield CompareGuessersReport()


class AllGuesses(WrapperTask):
    def requires(self):
        yield AllGuesserReports()
        yield GenerateAllGuesses()
