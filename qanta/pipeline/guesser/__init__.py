import os
import importlib
import pickle

import luigi
from luigi import LocalTarget, Task, WrapperTask

from qanta.pipeline.preprocess import Preprocess
from qanta.config import conf
from qanta.util import constants as c
from qanta.guesser.abstract import AbstractGuesser
from qanta.util.io import safe_path
from qanta import logging

log = logging.get(__name__)


def get_class(instance_module: str, instance_class: str):
    py_instance_module = importlib.import_module(instance_module)
    py_instance_class = getattr(py_instance_module, instance_class)
    return py_instance_class


def output_path(guesser_module: str, guesser_class: str, file: str):
    guesser_path = '{}.{}'.format(guesser_module, guesser_class)
    return safe_path(os.path.join(c.GUESSER_TARGET_PREFIX, guesser_path, file))


class EmptyTask(luigi.Task):
    def complete(self):
        return True


class TrainGuesser(Task):
    guesser_module = luigi.Parameter()  # type: str
    guesser_class = luigi.Parameter()  # type: str
    dependency_module = luigi.Parameter()  # type: str
    dependency_class = luigi.Parameter()  # type: str

    def requires(self):
        yield Preprocess()
        if self.dependency_class is not None and self.dependency_module is not None:
            dependency_class = get_class(self.dependency_module, self.dependency_class)
            yield dependency_class()

    def run(self):
        guesser_class = get_class(self.guesser_module, self.guesser_class)
        guesser_instance = guesser_class()  # type: AbstractGuesser
        qb_dataset = guesser_instance.qb_dataset()
        guesser_instance.train(qb_dataset.training_data())
        guesser_instance.save(output_path(self.guesser_module, self.guesser_class, ''))
        params = guesser_instance.parameters()
        params_path = output_path(self.guesser_module, self.guesser_class, 'guesser_params.pickle')
        with open(params_path, 'wb') as f:
            pickle.dump(params, f)

    def output(self):
        guesser_class = get_class(self.guesser_module, self.guesser_class)
        guesser_targets = [
            LocalTarget(file)
            for file in guesser_class.files(
                output_path(self.guesser_module, self.guesser_class, '')
            )]

        return [
            LocalTarget(output_path(self.guesser_module, self.guesser_class, '')),
            LocalTarget(
                output_path(self.guesser_module, self.guesser_class, 'guesser_params.pickle'))
        ] + guesser_targets


class GenerateGuesses(Task):
    guesser_module = luigi.Parameter()  # type: str
    guesser_class = luigi.Parameter()  # type: str
    dependency_module = luigi.Parameter()  # type: str
    dependency_class = luigi.Parameter()  # type: str
    word_skip = luigi.IntParameter(default=-1)  # type: int
    n_guesses = luigi.IntParameter(default=conf['n_guesses'])  # type: int

    def requires(self):
        yield TrainGuesser(
            guesser_module=self.guesser_module,
            guesser_class=self.guesser_class,
            dependency_module=self.dependency_module,
            dependency_class=self.dependency_class
        )

    def run(self):
        guesser_class = get_class(self.guesser_module, self.guesser_class)
        guesser_directory = output_path(self.guesser_module, self.guesser_class, '')
        guesser_instance = guesser_class.load(guesser_directory)  # type: AbstractGuesser

        for fold in c.ALL_FOLDS:
            log.info('Generating and saving guesses for {} fold'.format(fold))
            log.info('Starting guess generation...')
            guess_df = guesser_instance.generate_guesses(self.n_guesses, [fold],
                                                         word_skip=self.word_skip)
            log.info('Starting guess saving...')
            guesser_class.save_guesses(guess_df, guesser_directory, [fold])
            log.info('Done saving guesses')

    def output(self):
        targets = []
        for fold in c.ALL_FOLDS:
            targets.append(LocalTarget(output_path(
                self.guesser_module, self.guesser_class, 'guesses_{}.pickle'.format(fold))))
        return targets


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
            dependency_class=self.dependency_class
        )

    def run(self):
        guesser_class = get_class(self.guesser_module, self.guesser_class)
        guesser_directory = output_path(self.guesser_module, self.guesser_class, '')
        guesser_instance = guesser_class()
        guesser_instance.create_report(guesser_directory)

    def output(self):
        return [LocalTarget(output_path(
            self.guesser_module,
            self.guesser_class,
            'guesser_report.pdf')
        ), LocalTarget(output_path(
            self.guesser_module,
            self.guesser_class,
            'guesser_report.pickle'
        ))]


class AllGuessers(WrapperTask):
    def requires(self):
        guessers = conf['guessers']
        for g in guessers.values():
            if g['enabled']:
                guesser = g['class']
                dependency = g['luigi_dependency']
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

                yield GuesserReport(
                    guesser_module=guesser_module,
                    guesser_class=guesser_class,
                    dependency_module=dependency_module,
                    dependency_class=dependency_class
                )


class AllWordLevelGuesses(WrapperTask):
    def requires(self):
        guessers = conf['guessers']
        for g in guessers.values():
            if g['enabled']:
                guesser = g['class']
                dependency = g['luigi_dependency']
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

                yield GenerateGuesses(
                    guesser_module=guesser_module,
                    guesser_class=guesser_class,
                    dependency_module=dependency_module,
                    dependency_class=dependency_class,
                    word_skip=1,
                    n_guesses=25
                )
