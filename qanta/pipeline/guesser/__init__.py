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


class TrainGuesser(Task):
    guesser_module = luigi.Parameter()
    guesser_class = luigi.Parameter()
    dependency_module = luigi.Parameter()
    dependency_class = luigi.Parameter()

    def requires(self):
        dependency_class = get_class(self.dependency_module, self.dependency_class)
        yield Preprocess()
        yield dependency_class()

    def run(self):
        guesser_class = get_class(self.guesser_module, self.guesser_class)
        guesser_instance = guesser_class()  # type: AbstractGuesser
        datasets = guesser_instance.requested_datasets
        data = {}
        for name, dataset_instance in datasets.items():
            data[name] = dataset_instance.training_data()
        guesser_instance.train(data)
        guesser_instance.save(self._output_path(''))

    def output(self):
        guesser_class = get_class(self.guesser_module, self.guesser_class)
        guesser_targets = [
            LocalTarget(file)
            for file in guesser_class.files(self._output_path(''))]

        return [
            LocalTarget(self._output_path(''))
        ] + guesser_targets

    def _output_path(self, file):
        guesser_path = '{}.{}'.format(self.guesser_module, self.guesser_class)
        return safe_path(os.path.join(c.GUESSER_TARGET_PREFIX, guesser_path, file))


class AllGuessers(WrapperTask):
    def requires(self):
        for guesser, dependency in c.GUESSER_LIST:
            parts = guesser.split('.')
            guesser_module = '.'.join(parts[:-1])
            guesser_class = parts[-1]

            parts = dependency.split('.')
            dependency_module = '.'.join(parts[:-1])
            dependency_class = parts[-1]

            yield TrainGuesser(
                guesser_module=guesser_module,
                guesser_class=guesser_class,
                dependency_module=dependency_module,
                dependency_class=dependency_class
            )
