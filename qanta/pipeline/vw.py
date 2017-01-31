import luigi
from luigi import LocalTarget, Task, WrapperTask
from qanta.util import constants as c
from qanta.util.io import call, shell, make_dirs, safe_path
from qanta.reporting.vw_audit import parse_audit, audit_report
from qanta.pipeline.spark import SparkMergeFeatures


class VWMergeFeature(Task):
    fold = luigi.Parameter()

    def requires(self):
        return SparkMergeFeatures()

    def output(self):
        return [
            LocalTarget(
                c.VW_INPUT.format(self.fold)
            ),
            LocalTarget(c.VW_META.format(self.fold))
        ]

    def run(self):
        call(['bash', 'bin/vw_merge.sh', self.fold])


class VWMergeAllFeatures(WrapperTask):
    def requires(self):
        for fold in c.VW_FOLDS:
            yield VWMergeFeature(fold=fold)


class VWModel(Task):
    def requires(self):
        return VWMergeFeature(fold='dev')

    def output(self):
        return LocalTarget(c.VW_MODEL)

    def run(self):
        make_dirs('output/models/')
        call([
            'vw',
            '-d', c.VW_INPUT.format('dev'),
            '-b', '30',
            '--loss_function', 'logistic',
            '-f', c.VW_MODEL
        ])


class VWPredictions(Task):
    fold = luigi.Parameter()

    def requires(self):
        yield VWModel()
        yield VWMergeFeature(fold=self.fold)

    def output(self):
        make_dirs('output/predictions/')
        return LocalTarget(
            c.VW_PREDICTIONS.format(self.fold))

    def run(self):
        make_dirs('output/predictions/')
        call([
            'vw',
            '-t',
            '--loss_function', 'logistic',
            '-d', c.VW_INPUT.format(self.fold),
            '-i', c.VW_MODEL,
            '-p', c.VW_PREDICTIONS.format(self.fold)
        ])


class VWAudit(Task):
    def requires(self):
        yield VWModel()
        yield VWMergeFeature(fold='test')

    def output(self):
        return LocalTarget(
            c.VW_AUDIT.format('test')
        )

    def run(self):
        make_dirs('output/predictions/')
        shell(
            ('vw -t '
             '-d output/vw_input/{fold}.vw.txt '
             '--loss_function logistic '
             '-i {vw_model} --audit '
             '| python cli.py format_vw_audit '
             '> output/predictions/{fold}.audit').format(fold='test', vw_model=c.VW_MODEL)
        )


class VWAuditRegressor(Task):
    def requires(self):
        yield VWModel()

    def output(self):
        return [
            LocalTarget(safe_path(c.VW_AUDIT_REGRESSOR)),
            LocalTarget(safe_path(c.VW_AUDIT_REGRESSOR_REPORT))
        ]

    def run(self):
        call([
            'vw',
            '-t',
            '--loss_function', 'logistic',
            '-d', c.VW_INPUT.format('dev'),
            '-i', c.VW_MODEL,
            '--audit_regressor', c.VW_AUDIT_REGRESSOR
        ])
        df = parse_audit(c.VW_AUDIT_REGRESSOR)
        audit_report(df, c.VW_AUDIT_REGRESSOR_REPORT)
