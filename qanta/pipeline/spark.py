from itertools import product

import luigi
from luigi import LocalTarget, Task, WrapperTask
from clm.lm_wrapper import build_clm
from qanta.util import constants as c
from qanta.spark_execution import extract_features, merge_features
from qanta.pipeline.preprocess import Preprocess
from qanta.pipeline.dan import CreateGuesses
from qanta.util.classifier import build_classifier


class BuildClm(Task):
    def requires(self):
        yield Preprocess()

    def output(self):
        return LocalTarget(c.CLM_TARGET)

    def run(self):
        build_clm()


class ClassifierPickles(Task):
    class_type = luigi.Parameter()

    def requires(self):
        yield Preprocess()

    def output(self):
        return LocalTarget(c.CLASSIFIER_PICKLE_PATH.format(self.class_type))

    def run(self):
        build_classifier(self.class_type)


class AllClassifierPickles(WrapperTask):
    def requires(self):
        for t in c.CLASSIFIER_TYPES:
            yield ClassifierPickles(class_type=t)


class ExtractComputeFeatures(Task):
    resources = {'spark': 1}

    def requires(self):
        yield CreateGuesses()
        yield AllClassifierPickles()

    def output(self):
        targets = []
        for fold, feature in product(c.FOLDS, c.COMPUTE_OPT_FEATURES):
            targets.append(
                LocalTarget('output/features/{0}/sentence.{1}.parquet/'.format(fold, feature)))
        return targets

    def run(self):
        extract_features(c.COMPUTE_OPT_FEATURES)


class ExtractDeepFeatures(Task):
    resources = {'spark': 1}

    def requires(self):
        yield CreateGuesses()

    def output(self):
        targets = []
        for fold, feature in product(c.FOLDS, c.DEEP_OPT_FEATURES):
            targets.append(
                LocalTarget('output/features/{0}/sentence.{1}.parquet/'.format(fold, feature)))
        return targets

    def run(self):
        extract_features(c.DEEP_OPT_FEATURES)


class ExtractLMFeatures(Task):
    resources = {'spark': 1}

    def requires(self):
        yield CreateGuesses()
        yield BuildClm()

    def output(self):
        targets = []
        for fold, feature in product(c.FOLDS, c.LM_OPT_FEATURES):
            targets.append(
                LocalTarget('output/features/{0}/sentence.{1}.parquet/'.format(fold, feature)))
        return targets

    def run(self):
        extract_features(c.LM_OPT_FEATURES, lm_memory=True)


class ExtractMentionsFeatures(Task):
    resources = {'spark': 1}

    def requires(self):
        yield CreateGuesses()

    def output(self):
        targets = []
        for fold, feature in product(c.FOLDS, c.MENTIONS_OPT_FEATURES):
            targets.append(
                LocalTarget('output/features/{0}/sentence.{1}.parquet/'.format(fold, feature)))
        return targets

    def run(self):
        extract_features(c.MENTIONS_OPT_FEATURES, lm_memory=True)


class ExtractFeatures(WrapperTask):
    def requires(self):
        yield ExtractDeepFeatures()
        yield ExtractComputeFeatures()
        yield ExtractLMFeatures()
        yield ExtractMentionsFeatures()


class SparkMergeFeatures(Task):
    resources = {'spark': 1}

    def requires(self):
        return ExtractFeatures()

    def output(self):
        targets = []
        for fold, weight in product(c.FOLDS, c.NEGATIVE_WEIGHTS):
            targets.append(
                LocalTarget('output/vw_input/{0}/sentence.{1}.vw_input/'.format(fold, weight)))
        return targets

    def run(self):
        merge_features()
