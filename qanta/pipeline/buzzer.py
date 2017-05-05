import luigi
from luigi import LocalTarget, Task, WrapperTask
from qanta.util import constants as c
from qanta.util.io import call, shell, make_dirs, safe_path
from qanta.reporting.vw_audit import parse_audit, audit_report
from qanta.pipeline.spark import SparkMergeFeatures

BUZZER_MODEL = 'output/buzzer/mlp_buzzer.npz'

class BuzzerModel(Task):

    def output(self):
        return LocalTarget(BUZZER_MODEL)

    def run(self):
        make_dirs(safe_path('output/buzzers/'))
        shell(
            'python qanta/buzzer/cost_sensitive.py'
        )

class BuzzerBuzzes(Task):
    fold = luigi.Parameter()

    def requires(self):
        yield BuzzerModel()

    def output(self):
        return [
            LocalTarget(c.EXPO_BUZZ.format(self.fold)),
            LocalTarget(c.EXPO_FINAL.format(self.fold)),
            LocalTarget(c.VW_PREDICTIONS.format(self.fold)),
            LocalTarget(c.VW_META.format(self.fold))
            ]

    def run(self):
        make_dirs(safe_path('output/predictions/'))
        make_dirs(safe_path('output/expo/'))
        make_dirs(safe_path('output/vw_input/'))
        shell(
            'python qanta/buzzer/test.py -f {0}'.format(fold)
        )
