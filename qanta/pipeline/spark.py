from itertools import product

import luigi
from luigi import LocalTarget, Task, WrapperTask
from clm.lm_wrapper import build_clm
from qanta.util import constants as c
from qanta.spark_execution import extract_features, merge_features
from qanta.pipeline.preprocess import Preprocess
from qanta.pipeline.guesser import AllGuessers
from qanta.learning import classifier
from qanta.extractors.label import compute_question_stats
from qanta.util.environment import QB_QUESTION_DB
from qanta.util.qdb import QuestionDatabase


class BuildClm(Task):
    def requires(self):
        yield Preprocess()

    def output(self):
        return LocalTarget(c.CLM_TARGET)

    def run(self):
        build_clm()


class ClassifierPickle(Task):
    class_type = luigi.Parameter()

    def requires(self):
        yield Preprocess()

    def output(self):
        return LocalTarget(c.CLASSIFIER_PICKLE_PATH.format(self.class_type))

    def run(self):
        model = classifier.train_classifier(self.class_type)
        classifier.save_classifier(model, self.class_type)


class ClassifierReport(Task):
    class_type = luigi.Parameter()

    def requires(self):
        yield ClassifierPickle(class_type=self.class_type)

    def output(self):
        return LocalTarget(c.CLASSIFIER_REPORT_PATH.format(self.class_type))

    def run(self):
        model = classifier.load_classifier(self.class_type)
        classifier.create_report(model, self.class_type)


class AllClassifierPickles(WrapperTask):
    def requires(self):
        for t in c.CLASSIFIER_TYPES:
            yield ClassifierPickle(class_type=t)


class AllClassifierReports(WrapperTask):
    def requires(self):
        for t in c.CLASSIFIER_TYPES:
            yield ClassifierReport(class_type=t)


class AllClassifiers(WrapperTask):
    def requires(self):
        yield AllClassifierPickles()
        yield AllClassifierReports()


class ComputeParagraphStats(Task):
    def output(self):
        return LocalTarget(c.SENTENCE_STATS)

    def run(self):
        compute_question_stats(QB_QUESTION_DB)


class ExtractFastFeatures(Task):
    resources = {'spark': 1}

    def requires(self):
        yield AllGuessers()
        yield ComputeParagraphStats()

    def output(self):
        targets = []
        for fold, feature in product(c.VW_FOLDS, c.FAST_FEATURES):
            targets.append(
                LocalTarget('output/features/{0}/{1}.parquet/'.format(fold, feature)))
        return targets

    def run(self):
        extract_features(c.FAST_FEATURES)


class ExtractComputeFeatures(Task):
    resources = {'spark': 1}

    def requires(self):
        yield AllGuessers()
        yield AllClassifierPickles()

    def output(self):
        targets = []
        for fold, feature in product(c.VW_FOLDS, c.COMPUTE_OPT_FEATURES):
            targets.append(
                LocalTarget('output/features/{0}/{1}.parquet/'.format(fold, feature)))
        return targets

    def run(self):
        extract_features(c.COMPUTE_OPT_FEATURES)


class ExtractDeepFeatures(Task):
    resources = {'spark': 1}

    def requires(self):
        yield AllGuessers()

    def output(self):
        targets = []
        for fold, feature in product(c.VW_FOLDS, c.DEEP_OPT_FEATURES):
            targets.append(
                LocalTarget('output/features/{0}/{1}.parquet/'.format(fold, feature)))
        return targets

    def run(self):
        extract_features(c.DEEP_OPT_FEATURES)


class ExtractLMFeatures(Task):
    resources = {'spark': 1}

    def requires(self):
        yield AllGuessers()
        yield BuildClm()

    def output(self):
        targets = []
        for fold, feature in product(c.VW_FOLDS, c.LM_OPT_FEATURES):
            targets.append(
                LocalTarget('output/features/{0}/{1}.parquet/'.format(fold, feature)))
        return targets

    def run(self):
        extract_features(c.LM_OPT_FEATURES)


class ExtractMentionsFeatures(Task):
    resources = {'spark': 1}

    def requires(self):
        yield AllGuessers()

    def output(self):
        targets = []
        for fold, feature in product(c.VW_FOLDS, c.MENTIONS_OPT_FEATURES):
            targets.append(
                LocalTarget('output/features/{0}/{1}.parquet/'.format(fold, feature)))
        return targets

    def run(self):
        extract_features(c.MENTIONS_OPT_FEATURES)


class ExtractFeatures(WrapperTask):
    def requires(self):
        yield ExtractFastFeatures()
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
        for fold in c.VW_FOLDS:
            targets.append(LocalTarget('output/vw_input/{0}.vw/'.format(fold)))
        return targets

    def run(self):
        merge_features()
