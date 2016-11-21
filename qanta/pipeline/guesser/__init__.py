import os
import importlib

import luigi
from luigi import LocalTarget, Task, WrapperTask

from qanta.pipeline.preprocess import Preprocess
from qanta.util import constants as c


class TrainGuesser(Task):
    guesser_module = luigi.Parameter()
    guesser_class = luigi.Parameter()
    dependency_module = luigi.Parameter()
    dependency_class = luigi.Parameter()

    def requires(self):
        module = importlib.import_module(self.dependency_module)
        module_class = getattr(module, self.dependency_class)
        yield Preprocess()
        yield module_class()

    def run(self):
        module = importlib.import_module(self.guesser_module)
        module_class = getattr(module, self.guesser_class)
        guesser_instance = module_class()
        guesser_instance.train()
        guesser_path = '{}.{}'.format(self.guesser_module, self.guesser_class)
        guesser_instance.save(guesser_path)

    def output(self):
        guesser_path = '{}.{}'.format(self.guesser_module, self.guesser_class)
        return LocalTarget(os.path.join(c.GUESSER_TARGET_PREFIX, guesser_path))


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
