import luigi
from luigi import LocalTarget, Task, WrapperTask
from qanta.util import constants as c
from qanta.util.io import call, shell, make_dirs
from qanta.pipeline.vw import (VWPredictions, VWMergeFeature, VWAuditRegressor, VWAudit,
                               VWMergeAllFeatures)
from qanta.reporting import performance


@luigi.Task.event_handler(luigi.Event.PROCESSING_TIME)
def get_execution_time(self, processing_time):
    self.execution_time = processing_time


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


class ConcatReports(Task):
    resources = {'report_write': 1}

    def requires(self):
        yield VWAuditRegressor()
        yield AllSummaries()
        yield VWAudit()

    def output(self):
        return LocalTarget('output/reporting/report.pdf')

    def run(self):
        shell('pdftk output/reporting/*.pdf cat output /tmp/report.pdf')
        shell('mv /tmp/report.pdf output/reporting/report.pdf')


class FeatureAblation(Task):
    feature = luigi.Parameter()

    def requires(self):
        yield VWMergeAllFeatures()

    def output(self):
        return [
            LocalTarget('output/models/model.-{feature}.vw'.format(feature=self.feature)),
            LocalTarget('output/predictions/test.-{feature}.pred'.format(feature=self.feature)),
            LocalTarget('output/summary/test.-{feature}.json'.format(feature=self.feature))
        ]

    def run(self):
        shell('bin/feature-ablation.sh {file} {feature}'.format(file=self.feature,
                                                                feature=self.feature[0]))


class AllFeatureAblation(WrapperTask):
    def requires(self):
        for feature in c.FEATURE_NAMES:
            yield FeatureAblation(feature=feature)


class FeatureEval(Task):
    feature = luigi.Parameter()

    def requires(self):
        yield VWMergeAllFeatures()

    def output(self):
        return [
            LocalTarget('output/models/model.+{feature}.vw'.format(feature=self.feature)),
            LocalTarget('output/predictions/test.+{feature}.pred'.format(feature=self.feature)),
            LocalTarget('output/summary/test.+{feature}.json'.format(feature=self.feature))
        ]

    def run(self):
        shell('bin/feature-eval.sh {file} {feature}'.format(file=self.feature,
                                                            feature=self.feature[0]))


class AllFeatureEval(WrapperTask):
    def requires(self):
        for feature in c.FEATURE_NAMES:
            yield FeatureEval(feature=feature)


class PerformancePlot(Task):
    def requires(self):
        yield AllFeatureEval()
        yield AllFeatureAblation()
        yield Summary(fold='test')

    def output(self):
        return LocalTarget('output/summary/performance.png')

    def run(self):
        performance.plot_summary(False, 'output/summary/', 'output/summary/performance.png')


class All(WrapperTask):
    def requires(self):
        yield ConcatReports()
        yield AllFeatureAblation()
        yield AllFeatureEval()
        yield PerformancePlot()
