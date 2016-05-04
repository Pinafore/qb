from itertools import product
import subprocess
import luigi
from luigi import LocalTarget
from qanta.spark_execution import extract_features, merge_features
from qanta.util.constants import FOLDS, COMPUTE_OPT_FEATURES, MEMORY_OPT_FEATURES, NEGATIVE_WEIGHTS
from qanta.extract_features import create_guesses


def call(args):
    return subprocess.run(args, check=True)


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


class ExtractMemoryFeatures(luigi.Task):
    def requires(self):
        return ExtractComputeFeatures()

    def output(self):
        targets = []
        for fold, feature in product(FOLDS, MEMORY_OPT_FEATURES):
            targets.append(
                LocalTarget('data/features/{0}/sentence.{1}.parquet/'.format(fold, feature)))
        return targets

    def run(self):
        extract_features(MEMORY_OPT_FEATURES, lm_memory=True)


class ExtractFeatures(luigi.WrapperTask):
    def requires(self):
        yield ExtractComputeFeatures()
        yield ExtractMemoryFeatures()


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
        call(['bash', 'vw_merge.sh', self.fold, self.weight])


class VWMergeAllFeatures(luigi.WrapperTask):
    def requires(self):
        for fold, weight in product(FOLDS, NEGATIVE_WEIGHTS):
            yield VWMergeFeature(fold=fold, weight=str(weight))
