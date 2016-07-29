from itertools import product
import subprocess
import luigi
from luigi import LocalTarget
from qanta.spark_execution import extract_features, merge_features
from qanta.util import constants as C
from qanta.util import environment as E
from qanta.util.classifier import build_classifier
from qanta.extract_features import create_guesses
from qanta.guesser.util.format_dan import preprocess
from qanta.guesser.util import load_embeddings
from qanta.guesser import dan
from clm.lm_wrapper import build_clm


def call(args):
    return subprocess.run(args, check=True)


class GloveData(luigi.ExternalTask):
    def output(self):
        return LocalTarget("data/external/deep/glove.840B.300d.txt")


class FormatDan(luigi.Task):
    def requires(self):
        yield GloveData()

    def run(self):
        preprocess()

    def output(self):
        return [
            LocalTarget(C.DEEP_VOCAB_TARGET),
            LocalTarget(C.DEEP_TRAIN_TARGET),
            LocalTarget(C.DEEP_TEST_TARGET),
            LocalTarget(C.DEEP_DEV_TARGET),
            LocalTarget(C.DEEP_DEVTEST_TARGET)
        ]


class LoadEmbeddings(luigi.Task):
    def requires(self):
        yield FormatDan()

    def run(self):
        load_embeddings.create()

    def output(self):
        return LocalTarget(C.DEEP_WE_TARGET)


class TrainDAN(luigi.Task):
    def requires(self):
        yield FormatDan()
        yield LoadEmbeddings()

    def run(self):
        dan.train_dan()

    def output(self):
        return LocalTarget(C.DEEP_DAN_PARAMS_TARGET)


class ComputeDANOutput(luigi.Task):
    def requires(self):
        yield TrainDAN()
    
    def run(self):
        dan.compute_classifier_input()
    
    def output(self):
        return [
            LocalTarget(C.DEEP_DAN_TRAIN_OUTPUT),
            LocalTarget(C.DEEP_DAN_DEV_OUTPUT)
        ]


class TrainClassifier(luigi.Task):
    def requires(self):
        yield TrainDAN()
        yield ComputeDANOutput()

    def run(self):
        dan.train_classifier()

    def output(self):
        return LocalTarget(C.DEEP_DAN_CLASSIFIER_TARGET)


class BuildClm(luigi.Task):
    def output(self):
        return LocalTarget(C.CLM_TARGET)

    def run(self):
        build_clm()


class ClassifierPickles(luigi.Task):
    attribute = luigi.Parameter()

    def output(self):
        return LocalTarget('output/classifier/{0}.pkl'.format(self.attribute))

    def run(self):
        build_classifier(self.attribute, 'output/classifier/{0}.pkl'.format(self.attribute))


class AllClassifierPickles(luigi.WrapperTask):
    def requires(self):
        for t in C.CLASSIFIER_TYPES:
            yield ClassifierPickles(attribute=t)


class CreateGuesses(luigi.Task):
    def requires(self):
        yield TrainClassifier()

    def output(self):
        return LocalTarget(E.QB_GUESS_DB)

    def run(self):
        create_guesses(E.QB_GUESS_DB)


class ExtractComputeFeatures(luigi.Task):
    def requires(self):
        yield CreateGuesses()
        yield AllClassifierPickles()

    def output(self):
        targets = []
        for fold, feature in product(C.FOLDS, C.COMPUTE_OPT_FEATURES):
            targets.append(
                LocalTarget('output/features/{0}/sentence.{1}.parquet/'.format(fold, feature)))
        return targets

    def run(self):
        extract_features(C.COMPUTE_OPT_FEATURES)


class ExtractDeepFeatures(luigi.Task):
    def requires(self):
        yield ExtractComputeFeatures()

    def output(self):
        targets = []
        for fold, feature in product(C.FOLDS, C.DEEP_OPT_FEATURES):
            targets.append(
                LocalTarget('output/features/{0}/sentence.{1}.parquet/'.format(fold, feature)))
        return targets

    def run(self):
        extract_features(C.DEEP_OPT_FEATURES)


class ExtractLMFeatures(luigi.Task):
    def requires(self):
        yield ExtractComputeFeatures()
        yield ExtractDeepFeatures()
        yield BuildClm()

    def output(self):
        targets = []
        for fold, feature in product(C.FOLDS, C.LM_OPT_FEATURES):
            targets.append(
                LocalTarget('output/features/{0}/sentence.{1}.parquet/'.format(fold, feature)))
        return targets

    def run(self):
        extract_features(C.LM_OPT_FEATURES, lm_memory=True)


class ExtractMentionsFeatures(luigi.Task):
    def requires(self):
        yield ExtractComputeFeatures()
        yield ExtractDeepFeatures()
        yield ExtractLMFeatures()

    def output(self):
        targets = []
        for fold, feature in product(C.FOLDS, C.MENTIONS_OPT_FEATURES):
            targets.append(
                LocalTarget('output/features/{0}/sentence.{1}.parquet/'.format(fold, feature)))
        return targets

    def run(self):
        extract_features(C.MENTIONS_OPT_FEATURES, lm_memory=True)


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
        for fold, weight in product(C.FOLDS, C.NEGATIVE_WEIGHTS):
            targets.append(
                LocalTarget('output/vw_input/{0}/sentence.{1}.vw_input/'.format(fold, weight)))
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
            'output/vw_input/{0}.sentence.{1}.vw_input.gz'.format(self.fold, self.weight)),
                LocalTarget('output/vw_input/{0}.sentence.{1}.meta'.format(self.fold, self.weight)))

    def run(self):
        call(['bash', 'bin/vw_merge.sh', self.fold, str(self.weight)])


class VWMergeAllFeatures(luigi.WrapperTask):
    def requires(self):
        for fold, weight in product(C.FOLDS, C.NEGATIVE_WEIGHTS):
            yield VWMergeFeature(fold=fold, weight=weight)


class VWModel(luigi.Task):
    weight = luigi.IntParameter()

    def requires(self):
        return VWMergeFeature(fold='dev', weight=self.weight)

    def output(self):
        return LocalTarget('output/models/sentence.{0}.vw'.format(self.weight))

    def run(self):
        call([
            'vw',
            '--compressed',
            '-d', 'output/vw_input/dev.sentence.{0}.vw_input.gz'.format(self.weight),
            '--early_terminate', '100',
            '-k',
            '-q', 'ga',
            '-b', '28',
            '--loss_function', 'logistic',
            '-f', 'output/models/sentence.{0}.vw'.format(self.weight)
        ])


class VWPredictions(luigi.Task):
    fold = luigi.Parameter()
    weight = luigi.IntParameter()

    def requires(self):
        yield VWModel(weight=self.weight)
        yield VWMergeFeature(fold=self.fold, weight=self.weight)

    def output(self):
        return LocalTarget('output/predictions/{0}.sentence.{1}.pred'.format(self.fold, self.weight))

    def run(self):
        call([
            'vw',
            '--compressed',
            '-t',
            '-d', 'output/vw_input/{0}.sentence.{1}.vw_input.gz'.format(self.fold, self.weight),
            '-i', 'output/models/sentence.{0}.vw'.format(self.weight),
            '-p', 'output/predictions/{0}.sentence.{1}.pred'.format(self.fold, self.weight)
        ])


class VWSummary(luigi.Task):
    fold = luigi.Parameter()
    weight = luigi.IntParameter()

    def requires(self):
        yield VWMergeFeature(fold=self.fold, weight=self.weight)
        yield VWPredictions(fold=self.fold, weight=self.weight)

    def output(self):
        return LocalTarget('output/summary/{0}.sentence.{1}.json'.format(self.fold, self.weight))

    def run(self):
        call([
            'python3',
            'qanta/reporting/performance.py',
            'generate',
            'output/predictions/{0}.sentence.{1}.pred'.format(self.fold, self.weight),
            'output/vw_input/{0}.sentence.{1}.meta'.format(self.fold, self.weight),
            'output/summary/{0}.sentence.{1}.json'.format(self.fold, self.weight)
        ])


class AllSummaries(luigi.WrapperTask):
    def requires(self):
        for fold, weight in product(C.FOLDS, C.NEGATIVE_WEIGHTS):
            yield VWSummary(fold=fold, weight=weight)


class Ablation(luigi.Task):
    fold = luigi.Parameter()
    weight = luigi.IntParameter()
    feature = luigi.Parameter()

    def requires(self):
        yield VWMergeFeature(fold=self.fold, weight=self.weight)
