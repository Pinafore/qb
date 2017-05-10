import os
import luigi
from luigi import LocalTarget, Task, WrapperTask
from qanta.util import constants as c
from qanta.util.io import call, shell, make_dirs, safe_path
from qanta.reporting.vw_audit import parse_audit, audit_report
from qanta.pipeline.spark import SparkMergeFeatures
from qanta.guesser.abstract import AbstractGuesser
from qanta.buzzer import constants as bc

class MergeGuesserDFs(Task):

    def output(self):
        return [LocalTarget(AbstractGuesser.guess_path(bc.GUESSES_DIR, fold) for
            fold in c.BUZZ_FOLDS]
        
    def run(self):
        shell(
            'python qanta/buzzer/merge_dfs.py'
        )


class BuzzerModel(Task):

    def requires(self):
        yield MergeGuesserDFs()

    def output(self):
        return LocalTarget(bc.BUZZER_MODEL)

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
