import os
import importlib

import luigi
from luigi import LocalTarget, Task, WrapperTask

from qanta.pipeline.preprocess import Preprocess
from qanta.util import constants as c
from qanta.guesser.abstract import AbstractGuesser
from qanta.util.io import safe_path


def get_class(instance_module: str, instance_class: str):
    py_instance_module = importlib.import_module(instance_module)
    py_instance_class = getattr(py_instance_module, instance_class)
    return py_instance_class


def output_path(guesser_module: str, guesser_class: str, file):
    guesser_path = '{}.{}'.format(guesser_module, guesser_class)
    return safe_path(os.path.join(c.GUESSER_TARGET_PREFIX, guesser_path, file))


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

    def output(self):
        guesser_class = get_class(self.guesser_module, self.guesser_class)
        guesser_targets = [
            LocalTarget(file)
            for file in guesser_class.files(
                output_path(self.guesser_module, self.guesser_class, '')
            )]

        return [
            LocalTarget(output_path(self.guesser_module, self.guesser_class, ''))
        ] + guesser_targets


class GenerateGuesses(Task):
    guesser_module = luigi.Parameter()  # type: str
    guesser_class = luigi.Parameter()  # type: str
    dependency_module = luigi.Parameter()  # type: str
    dependency_class = luigi.Parameter()  # type: str

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

        guess_df = guesser_instance.generate_guesses(c.N_GUESSES, c.ALL_FOLDS)
        guesser_class.save_guesses(guess_df, guesser_directory)

    def output(self):
        return [
            LocalTarget(output_path(
                self.guesser_module, self.guesser_class, 'guesses_train.pickle')),
            LocalTarget(output_path(
                self.guesser_module, self.guesser_class, 'guesses_dev.pickle')),
            LocalTarget(output_path(
                self.guesser_module, self.guesser_class, 'guesses_test.pickle')),
            LocalTarget(output_path(
                self.guesser_module, self.guesser_class, 'guesses_devtest.pickle')),
        ]


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
        guesser_class.create_report(guesser_directory)

    def output(self):
        return LocalTarget(output_path(
            self.guesser_module,
            self.guesser_class,
            'guesser_report.pdf')
        )


class AllGuessers(WrapperTask):
    def requires(self):
        for guesser, dependency in c.GUESSER_LIST:
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
