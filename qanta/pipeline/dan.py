import os
import luigi
from luigi import LocalTarget, Task
from qanta.pipeline.preprocess import Preprocess
from qanta.util import constants as c
from qanta.util import environment as e
from qanta.guesser.util.format_dan import preprocess
from qanta.guesser.util import load_embeddings
from qanta.guesser import dan
from qanta.extract_features import create_guesses


class FormatDan(Task):
    def requires(self):
        yield Preprocess()

    def run(self):
        preprocess()

    def output(self):
        return [
            LocalTarget(c.DEEP_VOCAB_TARGET),
            LocalTarget(c.DEEP_TRAIN_TARGET),
            LocalTarget(c.DEEP_TEST_TARGET),
            LocalTarget(c.DEEP_DEV_TARGET),
            LocalTarget(c.DEEP_DEVTEST_TARGET)
        ]


class LoadEmbeddings(Task):
    def requires(self):
        yield FormatDan()

    def run(self):
        load_embeddings.create()

    def output(self):
        return LocalTarget(c.DEEP_WE_TARGET)


class TrainDAN(Task):
    def requires(self):
        yield LoadEmbeddings()

    def run(self):
        dan.train_dan()

    def output(self):
        return LocalTarget(c.DEEP_DAN_PARAMS_TARGET)


class ComputeDANOutput(Task):
    def requires(self):
        yield TrainDAN()

    def run(self):
        dan.compute_classifier_input()

    def output(self):
        return [
            LocalTarget(c.DEEP_DAN_TRAIN_OUTPUT),
            LocalTarget(c.DEEP_DAN_DEV_OUTPUT)
        ]


class TrainClassifier(Task):
    def requires(self):
        yield ComputeDANOutput()

    def run(self):
        dan.train_classifier()

    def output(self):
        return LocalTarget(c.DEEP_DAN_CLASSIFIER_TARGET)

class EvaluateClassifier(luigi.Task):
    def requires(self):
        yield TrainDAN()
        yield ComputeDANOutput()
        yield TrainClassifier()

    def run(self):
        dan.print_recall_at_n()

    def output(self):
        return LocalTarget(c.EVAL_RES_TARGET)

class CreateGuesses(Task):
    def requires(self):
        yield TrainClassifier()

    def output(self):
        return LocalTarget(e.QB_GUESS_DB)

    def run(self):
        create_guesses(e.QB_GUESS_DB)


@CreateGuesses.event_handler(luigi.Event.FAILURE)
def reset_guess_db(task, exception):
    os.remove(e.QB_GUESS_DB)
