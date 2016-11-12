from itertools import product

import luigi
from luigi import LocalTarget, Task, WrapperTask
from qanta.util import constants as c
from qanta.util.io import call, shell, make_dirs, safe_path
from qanta.reporting.vw_audit import parse_audit, audit_report
from qanta.pipeline.spark import SparkMergeFeatures


class VWMergeFeature(Task):
    fold = luigi.Parameter()
    weight = luigi.IntParameter()

    def requires(self):
        return SparkMergeFeatures()

    def output(self):
        return [
            LocalTarget(
                c.VW_INPUT.format(self.fold, self.weight)
            ),
            LocalTarget(c.VW_META.format(self.fold, self.weight))
        ]

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
        return LocalTarget(c.VW_MODEL.format(self.weight))

    def run(self):
        make_dirs('output/models')
        call([
            'vw',
            '--compressed',
            '-d', c.VW_INPUT.format('dev', self.weight),
            '--early_terminate', '100',
            '-k',
            '-q', 'ga',
            '-b', '28',
            '--loss_function', 'logistic',
            '-f', c.VW_MODEL.format(self.weight)
        ])


class VWPredictions(Task):
    fold = luigi.Parameter()
    weight = luigi.IntParameter()

    def requires(self):
        yield VWModel(weight=self.weight)
        yield VWMergeFeature(fold=self.fold, weight=self.weight)

    def output(self):
        return LocalTarget(
            c.VW_PREDICTIONS.format(self.fold, self.weight))

    def run(self):
        make_dirs('output/predictions')
        call([
            'vw',
            '--compressed',
            '-t',
            '--loss_function', 'logistic',
            '-d', c.VW_INPUT.format(self.fold, self.weight),
            '-i', c.VW_MODEL.format(self.weight),
            '-p', c.VW_PREDICTIONS.format(self.fold, self.weight)
        ])


class VWAudit(Task):
    weight = luigi.IntParameter()

    def requires(self):
        yield VWModel(weight=self.weight)
        yield VWMergeFeature(fold='test', weight=self.weight)

    def output(self):
        return LocalTarget(
            c.VW_AUDIT.format('test', self.weight)
        )

    def run(self):
        make_dirs('output/predictions')
        shell(
            ('vw --compressed -t '
             '-d output/vw_input/{fold}.sentence.{weight}.vw_input.gz '
             '-i output/models/sentence.{weight}.vw --audit '
             '| python cli.py format_vw_audit '
             '> output/predictions/{fold}.sentence.{weight}.audit').format(
                fold='test', weight=self.weight)
        )


class VWAuditRegressor(Task):
    weight = luigi.IntParameter()

    def requires(self):
        yield VWModel(weight=self.weight)

    def output(self):
        return [
            LocalTarget(safe_path(c.VW_AUDIT_REGRESSOR.format(self.weight))),
            LocalTarget(safe_path(c.VW_AUDIT_REGRESSOR_REPORT.format(self.weight)))
        ]

    def run(self):
        call([
            'vw',
            '--compressed',
            '-t',
            '--loss_function', 'logistic',
            '-d', c.VW_INPUT.format('dev', self.weight),
            '-i', c.VW_MODEL.format(self.weight),
            '--audit_regressor', c.VW_AUDIT_REGRESSOR.format(self.weight)
        ])
        df = parse_audit(c.VW_AUDIT_REGRESSOR.format(self.weight))
        audit_report(df, c.VW_AUDIT_REGRESSOR_REPORT.format(self.weight), self.weight)
