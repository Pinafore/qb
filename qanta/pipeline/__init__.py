import luigi
from luigi import LocalTarget, Task, WrapperTask
from qanta.util import constants as c
from qanta.util.io import call, shell, make_dirs
from qanta.pipeline.vw import VWPredictions, VWMergeFeature, VWAuditRegressor, VWAudit
from qanta.pipeline.preprocess import Preprocess


class Summary(Task):
    fold = luigi.Parameter()

    def requires(self):
        yield VWPredictions(fold=self.fold)

    def output(self):
        return LocalTarget('output/summary/{0}.json'.format(self.fold))

    def run(self):
        make_dirs('output/summary/')
        call([
            'python3',
            'qanta/reporting/performance.py',
            'generate',
            c.PRED_TARGET.format(self.fold),
            c.META_TARGET.format(self.fold),
            'output/summary/{0}.json'.format(self.fold)
        ])


class AllSummaries(WrapperTask):
    def requires(self):
        for fold in c.VW_FOLDS:
            yield Summary(fold=fold)


class Reports(Task):
    fold = luigi.Parameter()
    resources = {'report_write': 1}

    def requires(self):
        yield VWAuditRegressor()
        yield AllSummaries()

    def output(self):
        return LocalTarget('output/reporting/report.pdf')

    def run(self):
        shell('pdftk output/reporting/*.pdf cat output /tmp/report.pdf')
        shell('mv /tmp/report.pdf output/reporting/report.pdf')


class AllReports(WrapperTask):
    def requires(self):
        for fold in c.VW_FOLDS:
            yield Reports(fold=fold)


class AblationRun(Task):
    fold = luigi.Parameter()
    feature = luigi.Parameter()

    def requires(self):
        yield VWMergeFeature(fold=self.fold)
