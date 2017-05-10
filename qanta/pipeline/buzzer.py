import os
import luigi
from argparse import Namespace
from luigi import LocalTarget, Task, WrapperTask
from qanta.util import constants as c
from qanta.config import conf
from qanta.util.io import call, shell, make_dirs, safe_path
from qanta.reporting.vw_audit import parse_audit, audit_report
from qanta.guesser.abstract import AbstractGuesser
from qanta.buzzer import test as buzzer_test
from qanta.buzzer import constants as bc
from qanta.buzzer import configs as buzzer_configs
from qanta.buzzer.cost_sensitive import train_cost_sensitive
from qanta.buzzer.util import merge_dfs


class MergeGuesserDFs(Task):

    def output(self):
        return [LocalTarget(AbstractGuesser.guess_path(bc.GUESSES_DIR, fold)) \
                for fold in c.BUZZ_FOLDS]
        
    def run(self):
        merge_dfs()


class BuzzerModel(Task):

    def requires(self):
        yield MergeGuesserDFs()

    def output(self):
        cfg = getattr(buzzer_configs, conf['buzzer']['config'])
        return LocalTarget(cfg.model_dir)

    def run(self):
        make_dirs(safe_path('output/buzzers/'))
        args = Namespace(config=conf['buzzer']['config'], epochs=6, load=False)
        train_cost_sensitive(args)


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
        args = Namespace(fold=self.fold, config=conf['buzzer']['config'])
        buzzer_test.generate(args)

class AllBuzzes(WrapperTask):
    def requires(self):
        for fold in c.BUZZ_FOLDS:
            yield BuzzerBuzzes(fold=fold)
