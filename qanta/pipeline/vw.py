from itertools import product

import luigi
from luigi import LocalTarget, Task, WrapperTask
from qanta.util import constants as c
from qanta.pipeline.util import call
from qanta.pipeline.spark import SparkMergeFeatures


class VWMergeFeature(Task):
    fold = luigi.Parameter()
    weight = luigi.IntParameter()

    def requires(self):
        return SparkMergeFeatures()

    def output(self):
        return (LocalTarget(
            'output/vw_input/{0}.sentence.{1}.vw_input.gz'.format(self.fold, self.weight)),
                LocalTarget('output/vw_input/{0}.sentence.{1}.meta'.format(self.fold, self.weight)))

    def run(self):
        call(['bash', 'bin/vw_merge.sh', self.fold, str(self.weight)])


class VWMergeAllFeatures(WrapperTask):
    def requires(self):
        for fold, weight in product(c.FOLDS, c.NEGATIVE_WEIGHTS):
            yield VWMergeFeature(fold=fold, weight=weight)


class VWModel(Task):
    weight = luigi.IntParameter()

    def requires(self):
        return VWMergeFeature(fold='dev', weight=self.weight)

    def output(self):
        return LocalTarget('output/models/sentence.{0}.vw'.format(self.weight))

    def run(self):
        call([
            'vw',
            '--compressed',
            '-d', 'output/vw_input/dev.sentence.{0}.vw_input.gz'.format(self.weight),
            '--early_terminate', '100',
            '-k',
            '-q', 'ga',
            '-b', '28',
            '--loss_function', 'logistic',
            '-f', 'output/models/sentence.{0}.vw'.format(self.weight)
        ])


class VWPredictions(Task):
    fold = luigi.Parameter()
    weight = luigi.IntParameter()

    def requires(self):
        yield VWModel(weight=self.weight)
        yield VWMergeFeature(fold=self.fold, weight=self.weight)

    def output(self):
        return LocalTarget(
            'output/predictions/{0}.sentence.{1}.pred'.format(self.fold, self.weight))

    def run(self):
        call([
            'vw',
            '--compressed',
            '-t',
            '-d', 'output/vw_input/{0}.sentence.{1}.vw_input.gz'.format(self.fold, self.weight),
            '-i', 'output/models/sentence.{0}.vw'.format(self.weight),
            '-p', 'output/predictions/{0}.sentence.{1}.pred'.format(self.fold, self.weight)
        ])
