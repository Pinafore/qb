import luigi
from argparse import Namespace
from luigi import LocalTarget, Task, WrapperTask
from qanta.util import constants as c
from qanta.config import conf
from qanta.util.io import make_dirs, safe_path
from qanta.guesser.abstract import AbstractGuesser
from qanta.buzzer import test as buzzer_test
from qanta.buzzer import constants as bc
from qanta.buzzer import configs as buzzer_configs
from qanta.buzzer.cost_sensitive import train_cost_sensitive
from qanta.buzzer.util import merge_dfs


class MergeGuesserDFs(Task):

    def output(self):
        return [LocalTarget(AbstractGuesser.guess_path(bc.GUESSES_DIR, fold)) for fold in c.BUZZER_INPUT_FOLDS]
        
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
        train_cost_sensitive(conf['buzzer']['config'], c.BUZZER_GENERATION_FOLDS)


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
        config=conf['buzzer']['config']
        buzzer_test.generate(config, [self.fold])


class AllBuzzes(WrapperTask):
    def requires(self):
        for fold in c.BUZZER_GENERATION_FOLDS:
            yield BuzzerBuzzes(fold=fold)
