from itertools import product

import luigi
from luigi import LocalTarget, Task, WrapperTask
from qanta.util import constants as C
from qanta.pipeline.util import shell, call
from qanta.pipeline.vw import VWPredictions


class Summary(Task):
    fold = luigi.Parameter()
    weight = luigi.IntParameter()

    def requires(self):
        yield VWPredictions(fold=self.fold, weight=self.weight)

    def output(self):
        return LocalTarget('output/summary/{0}.sentence.{1}.json'.format(self.fold, self.weight))

    def run(self):
        call([
            'python3',
            'qanta/reporting/performance.py',
            'generate',
            'output/predictions/{0}.sentence.{1}.pred'.format(self.fold, self.weight),
            'output/vw_input/{0}.sentence.{1}.meta'.format(self.fold, self.weight),
            'output/summary/{0}.sentence.{1}.json'.format(self.fold, self.weight)
        ])


class AllSummaries(WrapperTask):
    def requires(self):
        for fold, weight in product(C.FOLDS, C.NEGATIVE_WEIGHTS):
            yield Summary(fold=fold, weight=weight)
