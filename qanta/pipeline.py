from itertools import product
import subprocess
import luigi
from luigi import LocalTarget
from qanta.spark_execution import extract_features, merge_features
from qanta.util.constants import (FOLDS, COMPUTE_OPT_FEATURES, DEEP_OPT_FEATURES,
                                  LM_OPT_FEATURES, MENTIONS_OPT_FEATURES, NEGATIVE_WEIGHTS)
from qanta.extract_features import create_guesses
from clm.lm_wrapper import build_clm


def call(args):
    return subprocess.run(args, check=True)


class BuildClm(luigi.Task):
    def output(self):
        return LocalTarget('data/language_model.txt')

    def run(self):
        build_clm()


class CreateGuesses(luigi.Task):
    def output(self):
        return LocalTarget('data/guesses.db')

    def run(self):
        create_guesses()


class ExtractComputeFeatures(luigi.Task):
    def requires(self):
        return CreateGuesses()

    def output(self):
        targets = []
        for fold, feature in product(FOLDS, COMPUTE_OPT_FEATURES):
            targets.append(
                LocalTarget('data/features/{0}/sentence.{1}.parquet/'.format(fold, feature)))
        return targets

    def run(self):
        extract_features(COMPUTE_OPT_FEATURES)


class ExtractDeepFeatures(luigi.Task):
    def requires(self):
        yield ExtractComputeFeatures()

    def output(self):
        targets = []
        for fold, feature in product(FOLDS, DEEP_OPT_FEATURES):
            targets.append(
                LocalTarget('data/features/{0}/sentence.{1}.parquet/'.format(fold, feature)))
        return targets

    def run(self):
        extract_features(DEEP_OPT_FEATURES)


class ExtractLMFeatures(luigi.Task):
    def requires(self):
        yield ExtractDeepFeatures()
        yield BuildClm()

    def output(self):
        targets = []
        for fold, feature in product(FOLDS, LM_OPT_FEATURES):
            targets.append(
                LocalTarget('data/features/{0}/sentence.{1}.parquet/'.format(fold, feature)))
        return targets

    def run(self):
        extract_features(LM_OPT_FEATURES, lm_memory=True)


class ExtractMentionsFeatures(luigi.Task):
    def requires(self):
        yield ExtractLMFeatures()

    def output(self):
        targets = []
        for fold, feature in product(FOLDS, MENTIONS_OPT_FEATURES):
            targets.append(
                LocalTarget('data/features/{0}/sentence.{1}.parquet/'.format(fold, feature)))
        return targets

    def run(self):
        extract_features(MENTIONS_OPT_FEATURES, lm_memory=True)


class ExtractFeatures(luigi.WrapperTask):
    def requires(self):
        yield ExtractDeepFeatures()
        yield ExtractComputeFeatures()
        yield ExtractLMFeatures()
        yield ExtractMentionsFeatures()


class SparkMergeFeatures(luigi.Task):
    def requires(self):
        return ExtractFeatures()

    def output(self):
        targets = []
        for fold, weight in product(FOLDS, NEGATIVE_WEIGHTS):
            targets.append(
                LocalTarget('data/vw_input/{0}/sentence.{1}.vw_input/'.format(fold, weight)))
        return targets

    def run(self):
        merge_features()


class VWMergeFeature(luigi.Task):
    fold = luigi.Parameter()
    weight = luigi.IntParameter()

    def requires(self):
        return SparkMergeFeatures()

    def output(self):
        return (LocalTarget(
            'data/vw_input/{0}.sentence.{1}.vw_input.gz'.format(self.fold, self.weight)),
                LocalTarget('data/vw_input/{0}.sentence.{1}.meta'.format(self.fold, self.weight)))

    def run(self):
        call(['bash', 'bin/vw_merge.sh', self.fold, str(self.weight)])


class VWMergeAllFeatures(luigi.WrapperTask):
    def requires(self):
        for fold, weight in product(FOLDS, NEGATIVE_WEIGHTS):
            yield VWMergeFeature(fold=fold, weight=weight)


class VWModel(luigi.Task):
    weight = luigi.IntParameter()

    def requires(self):
        return VWMergeFeature(fold='dev', weight=self.weight)

    def output(self):
        return LocalTarget('data/models/sentence.{0}.vw'.format(self.weight))

    def run(self):
        call([
            'vw',
            '--compressed',
            '-d', 'data/vw_input/dev.sentence.{0}.vw_input.gz'.format(self.weight),
            '--early_terminate', '100',
            '-k',
            '-q', 'ga',
            '-b', '28',
            '--loss_function', 'logistic',
            '-f', 'data/models/sentence.{0}.vw'.format(self.weight)
        ])


class VWPredictions(luigi.Task):
    fold = luigi.Parameter()
    weight = luigi.IntParameter()

    def requires(self):
        yield VWModel(weight=self.weight)
        yield VWMergeFeature(fold=self.fold, weight=self.weight)

    def output(self):
        return LocalTarget('data/predictions/{0}.sentence.{1}.pred'.format(self.fold, self.weight))

    def run(self):
        call([
            'vw',
            '--compressed',
            '-t',
            '-d', 'data/vw_input/{0}.sentence.{1}.vw_input.gz'.format(self.fold, self.weight),
            '-i', 'data/models/sentence.{0}.vw'.format(self.weight),
            '-p', 'data/predictions/{0}.sentence.{1}.pred'.format(self.fold, self.weight)
        ])


class VWSummary(luigi.Task):
    fold = luigi.Parameter()
    weight = luigi.IntParameter()

    def requires(self):
        yield VWMergeFeature(fold=self.fold, weight=self.weight)
        yield VWPredictions(fold=self.fold, weight=self.weight)

    def output(self):
        return LocalTarget('data/summary/{0}.sentence.{1}.json'.format(self.fold, self.weight))

    def run(self):
        call([
            'python3',
            'qanta/reporting/performance.py',
            'generate',
            'data/predictions/{0}.sentence.{1}.pred'.format(self.fold, self.weight),
            'data/vw_input/{0}.sentence.{1}.meta'.format(self.fold, self.weight),
            'data/summary/{0}.sentence.{1}.json'.format(self.fold, self.weight)
        ])


class AllSummaries(luigi.WrapperTask):
    def requires(self):
        for fold, weight in product(FOLDS, NEGATIVE_WEIGHTS):
            yield VWSummary(fold=fold, weight=weight)


class Ablation(luigi.Task):
    fold = luigi.Parameter()
    weight = luigi.IntParameter()
    feature = luigi.Parameter()

    def requires(self):
        yield VWMergeFeature(fold=self.fold, weight=self.weight)
