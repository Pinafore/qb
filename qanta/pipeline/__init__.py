from itertools import product
import os

import luigi
from luigi import LocalTarget, Task, WrapperTask
from qanta.util import constants as c
from qanta.util.io import call, shell
from qanta.pipeline.vw import VWPredictions, VWMergeFeature, VWAuditRegressor
from qanta.pipeline.dan import CreateGuesses, AllDAN
from qanta.pipeline.preprocess import Preprocess


class Summary(Task):
    fold = luigi.Parameter()
    weight = luigi.IntParameter()

    def requires(self):
        yield VWPredictions(fold=self.fold, weight=self.weight)

    def output(self):
        return LocalTarget('output/summary/{0}.sentence.{1}.json'.format(self.fold, self.weight))

    def run(self):
        if not os.path.exists('output/summary'):
            os.makedirs('output/summary')
        call([
            'python3',
            'qanta/reporting/performance.py',
            'generate',
            c.PRED_TARGET.format(self.fold, self.weight),
            c.META_TARGET.format(self.fold, self.weight),
            'output/summary/{0}.sentence.{1}.json'.format(self.fold, self.weight)
        ])


class Reports(Task):
    fold = luigi.Parameter()
    weight = luigi.IntParameter()

    resources = {'report_write': 1}

    def requires(self):
        yield VWAuditRegressor(weight=self.weight)
        yield Summary(fold=self.fold, weight=self.weight)

    def output(self):
        return LocalTarget('output/reporting/report.{}.pdf'.format(self.weight))

    def run(self):
        shell('pdftk output/reporting/*.pdf cat output /tmp/report.{}.pdf'.format(self.weight))
        shell('mv /tmp/report.{0}.pdf output/reporting/report.{0}.pdf'.format(self.weight))


class AllSummaries(WrapperTask):
    def requires(self):
        for fold, weight in product(c.FOLDS, c.NEGATIVE_WEIGHTS):
            yield Reports(fold=fold, weight=weight)


class AblationRun(Task):
    fold = luigi.Parameter()
    weight = luigi.IntParameter()
    feature = luigi.Parameter()

    def requires(self):
        yield VWMergeFeature(fold=self.fold, weight=self.weight)
